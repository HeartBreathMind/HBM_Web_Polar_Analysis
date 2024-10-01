import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Function to load and verify RR intervals from the uploaded file
def load_rr_intervals(file):
    rr_intervals = []
    for line in file:
        try:
            rr_intervals.append(float(line.strip()))
        except ValueError:
            st.error("File contains invalid data. Please upload a valid file with RR intervals.")
            return None
    return rr_intervals

def calculate_heart_rate(rr_intervals):
    time_beats = np.cumsum(rr_intervals)/1000
    heart_rates = [60000 /i for i in rr_intervals]
    time_seconds = np.arange(0, time_beats[-1], step=1)

    # Interpolate heart rate values at each second
    heart_rate_interpolator = interp1d(time_beats, heart_rates, bounds_error=False, fill_value="extrapolate")
    heart_rate_seconds = heart_rate_interpolator(time_seconds)
    return heart_rate_seconds  # HR in beats per minute

def calculate_rmssd(rr_intervals):
    rr_intervals = np.array(rr_intervals, dtype=np.float64)
    # Remove nan values
    rr_intervals = rr_intervals[~np.isnan(rr_intervals)]
    if len(rr_intervals) < 2:
        return np.nan
    rr_diff = np.diff(rr_intervals)
    rr_sqdiff = rr_diff ** 2
    rmssd = np.sqrt(np.nanmean(rr_sqdiff))
    return rmssd

def calculate_sdnn(rr_intervals):
    return np.std(rr_intervals)

def calculate_pnn50(rr_intervals):
    differences = np.abs(np.diff(rr_intervals))
    count_nn50 = np.sum(differences > 50)  # Number of successive intervals that differ by more than 50 ms
    pnn50 = (count_nn50 / len(differences)) * 100
    return pnn50

def calculate_sd1_sd2(rr_intervals):
    sd1 = np.std(np.diff(rr_intervals) / np.sqrt(2))
    sd2 = np.sqrt(2 * np.var(rr_intervals) - sd1**2)
    return sd1, sd2

def rolling_rmssd(rr_intervals, window_size=60, step_size=5):
    intervals = np.array(rr_intervals)
    # Convert RR intervals to R-peak times in seconds
    r_peaks_times = np.cumsum(intervals) / 1000.0  # Convert to seconds
    total_time = r_peaks_times[-1]
    if total_time < window_size:
        window_size = total_time
    # Generate time points for rolling calculation
    t0_list = np.arange(0, total_time - window_size + step_size, step_size)
    times = []
    rmssd_values = []
    for t0 in t0_list:
        t1 = t0 + window_size
        # Find indices of RR intervals within the window
        idx = np.where((r_peaks_times >= t0) & (r_peaks_times < t1))[0]
        if len(idx) > 1:
            # Extract the RR intervals in the window
            rr_win = intervals[idx]
            rmssd = calculate_rmssd(rr_win)
            rmssd_values.append(rmssd)
            times.append(t0 + window_size / 2.0)
        else:
            # Not enough data to compute RMSSD
            rmssd_values.append(np.nan)
            times.append(t0 + window_size / 2.0)
    return times, rmssd_values

def rolling_rmssd_summary(rmssd_values):
    rolling_rmssd_summary = {
        'mean': np.mean(rmssd_values),
        'min': np.nanpercentile(rmssd_values, 5),
        '25%': np.nanpercentile(rmssd_values, 25),
        '50%': np.nanpercentile(rmssd_values, 50),
        '75%': np.nanpercentile(rmssd_values, 75),
        'max': np.nanpercentile(rmssd_values, 95),
    }
    return rolling_rmssd_summary

def plot_rolling_hrv(rr_intervals, window_size=300, step_size=5):
    times, rmssd_values = rolling_rmssd(rr_intervals, window_size=300, step_size=5)
    rmssd_df = pd.DataFrame({'Time (s)': times, 'RMSSD (ms)': rmssd_values})    
    def seconds_to_english_time(seconds):
        if seconds < 60:
            # Less than 1 minute
            seconds = seconds
            parts = []
            if seconds > 0:
                parts.append(f"{seconds} Second")
            return ' '.join(parts)
        elif seconds < 3600:
            # Less than 1 hour
            minutes = seconds // 60
            seconds = seconds % 60
            parts = []
            if minutes > 0:
                parts.append(f"{minutes} Minute")
            if seconds > 0:
                parts.append(f"{seconds} Second")
            return ' '.join(parts)
        else:
            # 1 hour or more
            hours = seconds // 3600
            remainder = seconds % 3600
            minutes = remainder // 60
            parts = []
            if hours > 0:
                parts.append(f"{hours} Hour")
            if minutes > 0:
                parts.append(f"{minutes} Minute")
            return ' '.join(parts)
        
    fig = alt.Chart(rmssd_df).mark_line(point=True).encode(
        x=alt.X('Time (s)', title='Time (s)'),
        y=alt.Y('RMSSD (ms)', title='RMSSD (ms)'),
        tooltip=['Time (s)', 'RMSSD (ms)']
    ).properties(
        title= f'Rolling RMSSD Every {seconds_to_english_time(step_size)} Over {seconds_to_english_time(window_size)} Windows'
    ).interactive()

    # Display the chart in Streamlit
    st.title('Rolling RMSSD Over Time')
    st.altair_chart(fig, use_container_width=True)

def calculate_lf_hf_ratio(rr_intervals, fs):
    time = np.cumsum(rr_intervals) / 1000  # Convert RR intervals to seconds
    rr_interpolated = np.interp(np.arange(time[0], time[-1], 1/fs), time, rr_intervals)

    # Calculate power spectral density using Welch's method
    freqs, psd = welch(rr_interpolated, fs=fs, nperseg=len(rr_interpolated) // 2)
    
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    
    lf_power = np.trapezoid(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])], freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
    hf_power = np.trapezoid(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])], freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
    
    return lf_power, hf_power, lf_power / hf_power

def hr_chart(hr_data, age):
    max_hr = 220 - age
    zones = {
        'Below Zone 1': (0, 0.5 * max_hr),
        'Zone 1': (0.5 * max_hr, 0.6 * max_hr),
        'Zone 2': (0.6 * max_hr, 0.7 * max_hr),
        'Zone 3': (0.7 * max_hr, 0.8 * max_hr),
        'Zone 4': (0.8 * max_hr, 0.9 * max_hr),
        'Zone 5': (0.9 * max_hr, max_hr),
        'Above Zone 5': (max_hr, 220)
    }
    zone_colors = {
        'Below Zone 1': '#00B050',
        'Zone 1': '#FFE699',
        'Zone 2': '#F4B183',
        'Zone 3': '#C55A11',
        'Zone 4': '#F36A53',
        'Zone 5': '#FF0000',
        'Above Zone 5': '#C00000'
    }
    fig = px.line(hr_data, x='Time', y='heart_rate', title='Heart Rate Over Time', labels={'Time': 'Time', 'heart_rate': 'Heart Rate'})
    fig.update_traces(line=dict(color='blue'))
    fig.update_xaxes(tickformat="%H:%M:%S", linecolor='black', mirror=True)
    fig.update_yaxes(range=[0, 220], linecolor='black', mirror=True)
    shapes = []
    for zone_name, (low, high) in zones.items():
        shapes.append(go.layout.Shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, x1=1, y0=low, y1=high,
            fillcolor=zone_colors[zone_name], opacity=0.3,
            layer="below", line=dict(width=0)
        ))
    fig.update_layout(
        shapes=shapes,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )
    st.plotly_chart(fig)

def analyze_rr_session(rr_intervals, fs, lf_hf_thresh, acute_lf_hf_thresh, start_time, end_time, window_size=60, step_size=5):
    avg_hr = np.mean(calculate_heart_rate(rr_intervals))
    hr_range = f"{np.round(np.min(calculate_heart_rate(rr_intervals)), 0)} - {np.round(np.max(calculate_heart_rate(rr_intervals)), 0)}"
    sd1, sd2 = calculate_sd1_sd2(rr_intervals)
    rmssd = calculate_rmssd(rr_intervals)
    sdnn = calculate_sdnn(rr_intervals)
    pnn50 = calculate_pnn50(rr_intervals)
    lf_power, hf_power, lf_hf_ratio = calculate_lf_hf_ratio(rr_intervals, fs)
    rmssd_summary = rolling_rmssd_summary(rolling_rmssd(rr_intervals, window_size, step_size)[1])
    HRV_mean = rmssd_summary['mean']
    HRV_median = rmssd_summary['50%']
    HRV_range = f"{np.round(rmssd_summary['min'], 0)} - {np.round(rmssd_summary['max'], 0)}"
    
    # Determine sympathetic or parasympathetic state
    state = "Parasympathetic"
    if lf_hf_ratio > lf_hf_thresh:
        state = "Sympathetic"
    if lf_hf_ratio > acute_lf_hf_thresh:
        state = "Acute Sympathetic"
    results = {
        "Time": f"{start_time} - {end_time}",
        "Avg HR (bpm)": np.round(avg_hr, 0),
        "HR Range": hr_range,
        "HRV": np.round(rmssd, 0),
        "SDNN": np.round(sdnn, 0),
        "HRV_mean": np.round(HRV_mean, 0),         
        "HRV_median": np.round(HRV_median, 0), 
        "HRV_range": HRV_range,         
        "pNN50 (%)": np.round(pnn50, 1),
        "SD1": np.round(sd1, 0),
        "SD2": np.round(sd2, 0),
        "LF Power": np.round(lf_power, 0),
        "HF Power": np.round(hf_power, 0),
        "LF/HF Ratio": np.round(lf_hf_ratio, 1),
        "State": state
    }
    return results

def analyze_rr_chunks(rr_intervals, fs, lf_hf_thresh, acute_lf_hf_thresh, chunk_duration=None):
    results = []
    if chunk_duration:
        chunk_duration_secs = chunk_duration * 60  # Convert to seconds
        chunks = [(i, i+chunk_duration_secs) for i in range(0, int(sum(rr_intervals)/1000), chunk_duration_secs)]
        rri_time_df = pd.DataFrame({'RRI': rr_intervals, 'Time': pd.Series(rr_intervals).cumsum() / 1000})
        rri_time_df['time_hhmmss'] = pd.to_timedelta(rri_time_df['Time'], unit='s').apply(lambda x: str(x).split('.')[0].split(" ")[-1])
        rr_intervals_chunks = []
        for times in chunks:
            rr_intervals_chunks.append(rri_time_df[(rri_time_df.Time > times[0]) & (rri_time_df.Time <= times[1])].RRI.values.tolist())
        for i, chunk in enumerate(rr_intervals_chunks):
            results.append(analyze_rr_session(chunk, fs, lf_hf_thresh, acute_lf_hf_thresh, start_time=i * chunk_duration, end_time=int(np.ceil(min((i + 1) * chunk_duration, sum(rr_intervals)/60000)))))
    return results

def plot_hr_with_sympathetic(rr_intervals, results, chunks= False):
    hr = calculate_heart_rate(rr_intervals)
    time = np.arange(len(hr)) / 60  # Convert to minutes

    # Create a DataFrame to store heart rate and time
    data = pd.DataFrame({
        'Time (minutes)': time,
        'Heart Rate (bpm)': hr
    })

    # Create the base line chart for heart rate
    base = alt.Chart(data).mark_line(color='blue').encode(
        x='Time (minutes)',
        y='Heart Rate (bpm)'
    ).properties(
        width=700,
        height=400,
        title="Heart Rate with Sympathetic/Parasympathetic State"
    )
    if chunks:
    # Overlay shaded regions for sympathetic/parasympathetic states
        regions = []
        for result in results:
            start_time = int(result["Time"].split(" - ")[0])
            end_time = int(result["Time"].split(" - ")[1])
            
            if result["State"] == "Acute Sympathetic":
                color = 'red'
            elif result["State"] == "Sympathetic":
                color = 'orange'
            else:
                color = 'green'

            region = alt.Chart(pd.DataFrame({
                'start': [start_time],
                'end': [end_time]
            })).mark_rect(opacity=0.3, color=color).encode(
                x='start:Q',
                x2='end:Q'
            )
            regions.append(region)
        # Combine the heart rate line and state regions
        chart = base + alt.layer(*regions)
    else:
        chart = base

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

def openai_response(prompt, model = "gpt-4o"):
    client = OpenAI(api_key= openai_api_key.replace('"',''))
    response = client.chat.completions.create(
        model= model,
        messages=[ {"role": "user", "content": f"{prompt}"}]
        )
    return response.choices[0].message.content.strip()

# Streamlit app
def main():
    st.title("RR Interval Analysis Dashboard")

    # Step 1: File Upload
    uploaded_file = st.file_uploader("Upload a file containing RR intervals", type=["txt"])
    
    if uploaded_file is not None:
        rr_intervals = load_rr_intervals(uploaded_file)

        if rr_intervals is not None:
            st.success("File successfully uploaded and verified!")
            
            fs = 4  # Sampling frequency for interpolation
            lf_hf_thresh = 1.8
            acute_lf_hf_thresh = 3.0
            end_time = int(np.ceil(sum(rr_intervals)/60000))
            analysis_results = analyze_rr_session(rr_intervals, fs, lf_hf_thresh, acute_lf_hf_thresh, 0, end_time)
            st.write("Analysis Results:")
            for i,j in analysis_results.items(): st.write(f"{i}: {j}")
            plot_hr_with_sympathetic(rr_intervals, analysis_results, False)
            plot_rolling_hrv(rr_intervals, window_size= 300, step_size= 5)

            # Step 2: Get chunk duration from the user
            chunk_duration = st.number_input("Enter chunk duration (minutes)", min_value=1, max_value=60, value=5)

            if st.button("Run Analysis"):
                # Step 3: Perform analysis and plot results

                st.write("Performing analysis...")
                results = analyze_rr_chunks(rr_intervals, fs, lf_hf_thresh, acute_lf_hf_thresh, chunk_duration)

                st.write("Analysis Results:")

                # Step 4: Plot heart rate and sympathetic/parasympathetic states
                st.write("Heart Rate with Sympathetic/Parasympathetic State:")
                plot_hr_with_sympathetic(rr_intervals, results, True)
                for i, row in enumerate(results):
                    lf = row['LF Power']
                    hf = row['HF Power']
                    lf_hf_ratio = np.round(lf/hf, 2)
                    HRV_mean = row["HRV_mean"]
                    HRV_median = row["HRV_median"]
                    HRV_range = row["HRV_range"]                    
                    prompt = f"""
                    Here are the LF and HF and lf_hf_ratio from the Frequency Domain Analysis as well as the HRV_mean, HRV_median and HRV_range of the RR Intervals for a session: 
                    lf:{lf}, lf:{hf}, lf:{lf_hf_ratio}, lf:{HRV_mean}, lf:{HRV_median}, lf:{HRV_range}. 
                    Based on this data, please provide a layman-friendly explanation of what this means and whether they're in a relaxed or stressed situation and the impact this has on decision making. 
                    I define a sympathetic state as one with a lf/hf ratio above 1.8 and an LF/HF ratio above 3 to be in acute sympathetic state. 
                    If they're in a sympathetic state, give a quick tip on how to bring themselves back into a parasympathetic state. 
                    Respond in one paragraph, avoiding technical jargon and calling out any of the metrics and their values explicitly.
                    Do not explicitly call out the threshold values I provided. And address the reader as 'you'. Do not start with 'Based on the provided data', 'from the session data' or similar. 
                    """
                    for i,j in row.items():
                        st.write(f"{i}: {j}")
                    st.write(f"Analysis: \n{openai_response(prompt)}\n")
                
if __name__ == '__main__':
    main()

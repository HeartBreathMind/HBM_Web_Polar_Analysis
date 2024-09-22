import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d
import altair as alt

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

def calculate_pnn50(rr_intervals):
    differences = np.abs(np.diff(rr_intervals))
    count_nn50 = np.sum(differences > 50)  # Number of successive intervals that differ by more than 50 ms
    pnn50 = (count_nn50 / len(differences)) * 100
    return pnn50

def calculate_sd1_sd2(rr_intervals):
    sd1 = np.std(np.diff(rr_intervals) / np.sqrt(2))
    sd2 = np.sqrt(2 * np.var(rr_intervals) - sd1**2)
    return sd1, sd2

def calculate_rmssd(rr_intervals):
    diff_rr_intervals = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff_rr_intervals**2))

def calculate_sdnn(rr_intervals):
    return np.std(rr_intervals)

def analyze_rr_session(rr_intervals, fs, lf_hf_thresh, acute_lf_hf_thresh, start_time, end_time):
    avg_hr = np.mean(calculate_heart_rate(rr_intervals))
    hr_range = f"{np.round(np.min(calculate_heart_rate(rr_intervals)), 0)} - {np.round(np.max(calculate_heart_rate(rr_intervals)), 0)}"
    sd1, sd2 = calculate_sd1_sd2(rr_intervals)
    rmssd = calculate_rmssd(rr_intervals)
    sdnn = calculate_sdnn(rr_intervals)
    pnn50 = calculate_pnn50(rr_intervals)
    lf_power, hf_power, lf_hf_ratio = calculate_lf_hf_ratio(rr_intervals, fs)
    
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
        "SD1": np.round(sd1, 0),
        "SD2": np.round(sd2, 0),
        "RMSSD": np.round(rmssd, 0),
        "SDNN": np.round(sdnn, 0),
        "pNN50 (%)": np.round(pnn50, 1),
        "LF Power": np.round(lf_power, 0),
        "HF Power": np.round(hf_power, 0),
        "LF/HF Ratio": np.round(lf_hf_ratio, 1),
        "State": state
    }
    return results

def analyze_rr_chunks(rr_intervals, fs, lf_hf_thresh, acute_lf_hf_thresh, chunk_duration= None):
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
            results.append(analyze_rr_session(chunk, fs, lf_hf_thresh, acute_lf_hf_thresh, start_time= i * chunk_duration, end_time= int(np.ceil(min((i + 1) * chunk_duration, sum(rr_intervals)/60000)))))
    return results

def plot_hr_with_sympathetic(rr_intervals, results):
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

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

def session_analysis(file_path, chunk_duration= None, fs=4, lf_hf_thresh=1.8, acute_lf_hf_thresh=3., start_time= 0, end_time= None):
    rr_intervals = load_rr_intervals(file_path)
    if not end_time:
        end_time= sum(rr_intervals)/60000
    analysis_results = analyze_rr_session(rr_intervals, fs, lf_hf_thresh, acute_lf_hf_thresh, start_time, end_time)
    print(f"Full session analysis: \n{analysis_results}\n")
    if chunk_duration:
        chunk_analysis = analyze_rr_chunks(rr_intervals, fs, lf_hf_thresh, acute_lf_hf_thresh, chunk_duration=chunk_duration)
        print(f"Analysis of chunks analysis: \n{pd.DataFrame(chunk_analysis)}\n")
        plot_hr_with_sympathetic(rr_intervals, chunk_analysis)

# Streamlit app
def main():
    st.title("RR Interval Analysis Dashboard")

    # Step 1: File Upload
    uploaded_file = st.file_uploader("Upload a file containing RR intervals", type=["txt"])
    
    if uploaded_file is not None:
        rr_intervals = load_rr_intervals(uploaded_file)

        if rr_intervals is not None:
            st.success("File successfully uploaded and verified!")
            
            # Step 2: Get chunk duration from the user
            chunk_duration = st.number_input("Enter chunk duration (minutes)", min_value=1, max_value=60, value=5)

            if st.button("Run Analysis"):
                # Step 3: Perform analysis and plot results
                fs = 4  # Sampling frequency for interpolation
                lf_hf_thresh = 1.8
                acute_lf_hf_thresh = 3.0

                st.write("Performing analysis...")
                results = analyze_rr_chunks(rr_intervals, fs, lf_hf_thresh, acute_lf_hf_thresh, chunk_duration)

                st.write("Analysis Results:")
                st.write(pd.DataFrame(results))

                # Step 4: Plot heart rate and sympathetic/parasympathetic states
                st.write("Heart Rate with Sympathetic/Parasympathetic State:")
                plot_hr_with_sympathetic(rr_intervals, results)
                
if __name__ == '__main__':
    main()

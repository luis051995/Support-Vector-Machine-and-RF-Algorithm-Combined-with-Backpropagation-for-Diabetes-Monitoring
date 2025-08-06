# Support-Vector-Machine-and-RF-Algorithm-Combined-with-Backpropagation-for-Diabetes-Monitoring
This script presents a hybrid system for diabetes prediction. It combines SVM for non-linear pattern detection, Random Forest for robustness and feature importance, and a Neural Network to integrate both outputs via backpropagation. The model ensures high accuracy for healthcare use.

Code Explanation: Calibration Curve for Acetone with MQ-138 Sensor

This script performs the calibration of the MQ-138 sensor to measure acetone concentration in parts per million (PPM) based on the Rs/R0 ratio (sensor resistance over baseline resistance). Below are the key steps:

    Input Data:
    Two NumPy arrays are defined:

        ppm: known acetone concentrations in PPM.

        rs_r0: corresponding measured Rs/R0 values for each concentration.

    Data Display:
    The PPM and Rs/R0 pairs are printed to the console for verification.

    Logarithmic Transformation:
    To facilitate linear fitting, base-10 logarithms are applied to both arrays (ppm and rs_r0). This is common because the sensor’s response often follows a power-law relationship.

    Linear Fit (Regression):
    np.polyfit is used to fit a straight line (degree 1 polynomial) to the log-transformed data. The output is the slope m and intercept b of the line modeling the logarithmic relationship.

    Plotting:

        Points for the fitted line (x_fit, y_fit) are generated.

        A log-log plot is created showing the original data points and the fitted line.

        Labels, title, legend, and grid are added for clarity.

        The plot is saved as a high-resolution PNG file.

    Result:

        The plot shows how Rs/R0 changes with acetone concentration on a logarithmic scale.

        The fit parameters allow converting an Rs/R0 reading into an estimated acetone concentration.

Code Explanation: Real-time Acetone PPM Reading from MQ-138 Sensor via Serial Port

This script reads sensor voltage data from an MQ-138 acetone sensor connected through a serial port (e.g., Arduino or ESP32) and calculates the acetone concentration in parts per million (PPM) in real time.
Key components:

    Serial communication setup:

        The script opens a serial port (COM3) with a baud rate of 9600 to receive sensor data.

        A timeout of 2 seconds is set to avoid blocking indefinitely.

    Constants used for calculation:

        RL: Load resistance in kilo-ohms (10 kΩ).

        m and b: Calibration parameters obtained from previous curve fitting.

        R0: Baseline sensor resistance in clean air (10 kΩ).

    Reading voltage data:

        The function read_voltage() reads a line from the serial port, decodes it to a string, and tries to convert it to a floating-point number representing voltage.

        Invalid or non-positive values are ignored.

    Calculating sensor resistance (Rs):

        Using the formula:
        Rs=Vsupply×RLVsensor−RL
        Rs​=Vsensor​Vsupply​×RL​​−RL​

        where VsupplyVsupply​ is 3.3 V, RLRL​ is the load resistance, and VsensorVsensor​ is the measured voltage.

    Calculating ratio and PPM:

        The ratio Rs/R0Rs/R0 is computed.

        Using the logarithmic calibration parameters m and b, the acetone concentration (PPM) is calculated with the inverse of the fitted curve formula:
        PPM=10log⁡10(Rs/R0)−bm
        PPM=10mlog10​(Rs/R0)−b​

    Continuous reading loop:

        The program continuously reads voltage, calculates PPM, and prints the result every second.

        The loop can be stopped by pressing Ctrl+C.

    Resource cleanup:

        When interrupted, the serial port is safely closed.

Purpose

This code provides real-time monitoring of acetone gas concentration from sensor voltage measurements. It is suitable for applications like breath analysis, environmental monitoring, or safety systems where continuous gas concentration readings are needed.

Code Explanation: Real-Time Diabetes Risk Monitoring with Sensor Data, Machine Learning, and Flask API

This Python script performs real-time monitoring of acetone levels (PPM) using sensor data from an MQ-138 gas sensor connected via a serial port (e.g., Arduino/ESP32). It integrates machine learning models (SVM and Neural Network) for diabetes risk inference and exposes the results via a REST API built with Flask.
Main Components:

    Flask API Setup:

        A Flask web server runs on 0.0.0.0:5000, with CORS enabled.

        It serves an endpoint /resultado that returns the latest inference results as JSON.

        The API runs in a separate background thread to allow simultaneous sensor reading and web serving.

    Model Loading:

        The script loads a pre-trained scaler (for feature normalization), an SVM model (svm_model), and a Neural Network (nn_model) from saved files.

    User Input:

        At startup, the user inputs personal parameters: age, weight, height, sex, and medical history (antecedents).

        These inputs serve as features for the machine learning inference alongside sensor readings.

    Serial Port Communication:

        Opens serial connection on COM3 at 9600 baud rate.

        Continuously reads voltage values from the sensor.

        Performs sensor calibration to estimate R0 by averaging initial readings.

        Measures baseline acetone concentration (PPM) to stabilize measurements.

    PPM and Slope Calculation:

        Converts voltage readings to sensor resistance (RS), then computes the Rs/R0 ratio.

        Uses the calibration curve (parameters m and b) to calculate acetone concentration in PPM.

        Calculates the slope (rate of change) of PPM readings over time to detect sudden increases (like a person blowing on the sensor).

    Machine Learning Inference on Slope Detection:

        If the slope exceeds a threshold (slope_threshold), the system:

            Constructs an input feature vector combining sensor PPM and user info.

            Normalizes features with the scaler.

            Uses the SVM model to compute a decision score and probability.

            Feeds combined inputs into the Neural Network to get a final risk output.

        Prints the inference scores and risk status.

        Updates a global dictionary resultado_inferencia with the latest readings and predictions for API access.

    Error Handling and Cleanup:

        Runs continuously until interrupted.

        Prints warnings on exceptions without stopping the program.

Summary

This script integrates sensor data acquisition, real-time processing, and advanced ML inference into a cohesive system for diabetes risk monitoring based on breath acetone levels. The Flask API allows external applications (like a mobile app built with Ionic) to query the latest results instantly.
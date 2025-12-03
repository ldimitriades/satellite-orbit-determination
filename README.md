# GPS-Based Navigation and State Estimation for GRACE-A Using Pseudorange Data

## Overview
This project uses real GRACE-A GPS pseudorange measurements to estimate the spacecraft’s position by applying both a batch least-squares estimator and a sequential Kalman filter written from scratch. The workflow includes modeling and correcting key measurement errors such as receiver clock offsets, signal light-time delays, and relativistic effects. The Kalman filter implementation provides improved robustness and real-time estimation capability compared to the batch method. The project demonstrates end-to-end spacecraft navigation processing, from raw measurements to filtered state estimates.

## Key Concepts
-GPS pseudorange measurement modeling
-Least-squares position estimation
-Receiver clock error estimation
-Light-time correction
-Relativistic range correction
-Kalman filter design and implementation
-Sequential estimation vs. batch estimation
-Real-time navigation filtering
-Measurement residual analysis

## What I Did
-Processed raw GRACE-A pseudorange measurements provided for the project
-Implemented a full measurement model including receiver clock offsets, light-time delay, and relativistic corrections
-Developed a batch least-squares estimator to compute spacecraft position and clock bias
-Wrote a Kalman filter from scratch (state propagation, measurement update, covariance update)
-Tuned process and measurement noise models to achieve stable filtering performance
-Compared position accuracy, convergence, and stability between least-squares and Kalman filter approaches
-Generated residual plots and estimation error comparisons to evaluate estimator performance

## How to run
python Data_Assignment_2/least_squares.py
python Data_Assignment_3_with_corrections_applied/kalman_filter.py

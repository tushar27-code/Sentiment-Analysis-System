\# Sentiment Analysis System – Customers Feedback Analytics



A Python-based web application that analyzes customer feedback and classifies opinions into Positive, Negative, and Neutral sentiments using VADER.



\## Features



\- Upload customer feedback datasets using CSV files.

\- Load data from a Google Sheets CSV link.

\- Perform automated sentiment analysis using VADER.

\- Calculate Positive, Negative, Neutral, and Compound scores.

\- Interactive dashboard with sentiment charts and statistics.

\- Gender-wise sentiment analysis when Gender data is available.

\- Search feedback using keywords.

\- Filter records by sentiment category.

\- Identify negative customer feedback.

\- Display the top 10 frequent words in negative feedback.

\- Export analyzed results as CSV.

\- Generate downloadable PDF reports.



\## Technologies Used



\- Python

\- Streamlit

\- Pandas

\- VADER Sentiment

\- Plotly Express

\- ReportLab



\## Dataset Requirements



The dataset should be in CSV format and contain:



\- `Opinion` – Required column containing customer feedback.

\- `Gender` – Optional column for gender-wise analysis.



\## Installation



Install the required Python libraries:



```bash

pip install streamlit pandas vaderSentiment plotly reportlab

```



\## Run the Application



Open the project directory in CMD and run:



```bash

streamlit run main.py

```



The application will open in your web browser.



\## Project Files



| File | Description |

|---|---|

| `main.py` | Main Streamlit application |

| `practice.py` | Sentiment analysis practice script |

| `Sentiment\_Analysis\_System\_Project\_Report.pdf` | Project documentation |

| `.gitignore` | Files excluded from Git tracking |



\## Sentiment Classification



The system uses VADER compound scores:



\- Positive: Compound score >= 0.05

\- Negative: Compound score <= -0.05

\- Neutral: Compound score between -0.05 and 0.05



\## Future Enhancements



\- Multilingual sentiment analysis.

\- Advanced topic extraction.

\- Sentiment trends over time.

\- Database integration.

\- Improved analytics and reporting.



\## Author



Developed as a Python-based customer feedback analytics project.


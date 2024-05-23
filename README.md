# TSE-ASSIST
DataLoader for Tehran Stock Exchange (TSE) Data
This Python script provides a comprehensive tool to download, process, and analyze data from the Tehran Stock Exchange (TSE). It leverages the pytse-client library to retrieve historical and client type data for various stocks. The script is designed to support extensive data handling and extraction functionalities, making it ideal for researchers and analysts working with TSE data.
Key Features:
Download TSE Data: The script can download both historical ticker data and client type data for specified stock symbols.
Data Processing: It processes and extracts relevant data fields, handles periods for comparison, and calculates relative metrics.
Excel Export: The processed data is exported to Excel files for further analysis or reporting.
Dependencies:
os: For directory and file operations.
sys: For system-level operations, specifically to suppress and restore standard output.
glob: For file pattern matching.
tqdm: For displaying progress bars during data download operations.
logging: For logging warnings and errors.
numpy: For numerical operations.
pandas: For data manipulation and analysis.
pytse-client: For fetching TSE data.
Script Breakdown:
Initialization:

Class: DataLoader
Parameters:
symbols: List of stock symbols or 'all' to fetch all symbols.
fileName: Name of the output Excel file (default: 'output').
typeData: Type of data to fetch ('client', 'history', 'client&history').
periods: List of periods for data comparison.
Functionality: Initializes parameters, creates necessary directories, validates inputs, and sets up logging.
Data Download:

Functions:
downloadCTD(): Downloads client type data.
downloadHTD(): Downloads historical ticker data.
Progress Bars: Uses tqdm to show download progress.
Data Extraction and Processing:

Functions:
extractDataCTD(): Extracts and processes client type data.
extractDataHTD(): Extracts and processes historical ticker data.
dataScreamingCTD(): Validates and screens client type data.
dataScreamingHTD(): Validates and screens historical ticker data.
Data Wrangling: Merges data from different periods and calculates relative metrics.
Excel Export:

Function:
Saves processed data to Excel files for each period and data type.
Command Line Interface:

Options:
Download data.
Process and analyze data.
Exit the script.
User Interaction: Takes user input to specify periods and filenames for data processing.
How to Use:
Setup:

Ensure all dependencies are installed: pip install os sys glob tqdm logging numpy pandas pytse-client.
Place the script in a working directory with write permissions.
Execution:

Run the script: python script_name.py.
Follow on-screen instructions to download data and specify processing parameters.
Output:

Processed data will be saved in the specified directory in Excel format.


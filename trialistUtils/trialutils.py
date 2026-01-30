import re
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import base64

def extract_json_substring(text):
    match = re.search(r'```json(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else None

def save_webpage_as_pdf(url, output_pdf):
    # Set up Chrome options
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
    chrome_options.add_argument("--no-sandbox")  # Needed for some environments
    chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resources

    # Initialize WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Open the webpage
        driver.get(url)

        # Print the page to a PDF
        print_options = {
            "landscape": False,  # Set to True if you want landscape orientation
            "displayHeaderFooter": False,
            "printBackground": True,  # Print background graphics
            "preferCSSPageSize": True
        }
        pdf_data = driver.execute_cdp_cmd("Page.printToPDF", print_options)

        # print(f'pdf_data["data"] len: {len(pdf_data["data"])}, value:', pdf_data["data"])

        # Save the PDF file
        with open(output_pdf, "wb") as f:
            f.write(base64.b64decode(pdf_data["data"]))

        print(f"Webpage saved as {output_pdf}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        driver.quit()  # Close the browser

    return
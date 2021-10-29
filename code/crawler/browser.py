from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import wget
import re


options = Options()
options.add_argument("start-maximized"); # open Browser in maximized mode
options.add_argument("disable-infobars"); # disabling infobars
options.add_argument("--disable-extensions"); # disabling extensions
options.add_argument("--disable-dev-shm-usage"); # overcome limited resource problems
options.add_argument("--no-sandbox"); # Bypass OS security model


options.add_argument("user-data-dir=selenium")

driver = webdriver.Chrome('./chromedriver', chrome_options=options)

try:
	for w in range(13, 16):
		driver.get("https://www.coursera.org/learn/cs-410/home/week/" + str(w))
		time.sleep(5)
		try:
			lessons = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH, '//div[child::h3[contains(text(), "Lessons")]]/following-sibling::ul/li/a')))
		except Exception as e:
			print('Could not find lessons for week: ' + str(w) + '. Error: ' + str(e))
		else: 
			for l in range(len(lessons)):
				WebDriverWait(driver, 30).until(
					EC.presence_of_all_elements_located((By.XPATH, '//div[child::h3[contains(text(), "Lessons")]]/following-sibling::ul/li/a')))[l].click()
				lesson_name = WebDriverWait(driver, 10).until(
					EC.presence_of_element_located((By.XPATH, '//h2[contains(@class, "video-name")]'))).text
				lesson_name = re.sub('\\s+', '_', lesson_name.replace(':', ' ').replace('.', ' ').replace('-', ' '))
				WebDriverWait(driver, 10).until(
					EC.presence_of_element_located((By.XPATH, '//button[child::span[contains(text(), "Download")]]'))).click()
				try: 
					transcript_href = WebDriverWait(driver, 10).until(
						EC.presence_of_element_located((By.XPATH, '//button[child::span[contains(text(), "Download")]]/following-sibling::ul/li/a[descendant::span[contains(text(), "Transcript")]]'))).get_attribute("href")
					wget.download(transcript_href, out="../../data/transcripts/" + lesson_name + ".txt")
					time.sleep(10)
				except:
					print('No transcripts found for week: ' + str(w) + ", lecture: " + str(l + 1))	
				try:
					slides_href = WebDriverWait(driver, 10).until(
						EC.presence_of_element_located((By.XPATH, '//button[child::span[contains(text(), "Download")]]/following-sibling::ul/li/a[descendant::span[contains(text(), "Lecture Slides")]]'))).get_attribute("href")
					wget.download(transcript_href, out="../../data/slides/" + lesson_name + ".pdf")
					time.sleep(10)
				except:
					print('No slides found for week: ' + str(w) + ", lecture: " + str(l + 1))	
				driver.execute_script("window.history.go(-1)")
				time.sleep(5)
finally:
	driver.close()
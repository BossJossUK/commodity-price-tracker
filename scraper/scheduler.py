import schedule
import time
import logging
from scraper import run_scraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scheduler")

def job():
    """
    Run the scraper job and handle any exceptions
    """
    try:
        logger.info("Running scheduled scraper job")
        run_scraper()
        logger.info("Scheduled job completed successfully")
    except Exception as e:
        logger.error(f"Error in scheduled job: {str(e)}")

def main():
    """
    Setup schedule and run the scheduler
    """
    logger.info("Starting scheduler")
    
    # Schedule the job to run daily at specific times
    schedule.every().day.at("09:00").do(job)
    schedule.every().day.at("12:00").do(job)
    schedule.every().day.at("16:30").do(job)
    
    # Also run immediately on startup
    logger.info("Running initial scrape job")
    job()
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()

import sys
import logging



class CustomException(Exception):
    def __init__(self, error, error_detail=None):
        super().__init__(error)
        if error_detail:
            _, _, exc_tb = sys.exc_info()
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            self.error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{error}]"
        else:
            self.error_message = error

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(e, sys)  # Pass 'sys' only if you need to include file and line number details


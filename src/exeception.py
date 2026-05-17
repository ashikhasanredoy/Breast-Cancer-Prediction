import sys
from src.logger import logging

def error_message_details(error,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    line_num=exc_tb.tb_lineno
    
    
    error_message='error occured in the python cripted[{0}] line number [{1}] error massage[{2}]'.format(
     file_name,
     line_num,
     str(error)
    )
    return error_message


class CustomBreasctCancerException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        
        self.error_message=error_message_details(
            error_message,
            error_details=error_details
        )
    
    def __str__(self):
        return self.error_message    
    
logging.info("custom exception is done succesfully.")    


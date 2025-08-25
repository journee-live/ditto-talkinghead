class UnsupportedSourceException(RuntimeError):
   """Exception raised when an unsupported data source is encountered.
   
   This exception is raised when attempting to process or interact with
   a data source that is not supported by the current implementation.
   """
   
   def __init__(self, message: str ="Unsupported data source", source: str=""):
       """Initialize the exception.
       
       Args:
           message (str): Error message describing the exception
           source (str, optional): The unsupported source identifier
       """
       if source:
           message = f"{message}: {source}"
       super().__init__(message)
       self.source = source
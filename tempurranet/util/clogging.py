import datetime


class GeneralLogger:

    def __init__(self, logger_name: str):
        self.logger_name = logger_name

    def __log(self, message: any, type_log: str, end: str = '', flush: bool = False, pre: str = '', post: any = '',
              send: bool = True) -> None:
        """
        Logs text.
        :param message: Message to be logged.
        :param type_log: Type of log. i.e. err, info, warn
        :param end: End text of message
        :param flush: Clear or not clear current line
        :param pre: Text to add before string
        :param post: Text to add after the log name
        :param send: Determines whether to log the message.
        """
        if send:
            print(f"{pre}[{self.logger_name}/{type_log}] {post + ' ' if post != '' else ''}"
                  f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                  f"{message}",
                  end=end, flush=flush)

    def info(self, message: any, end: str = '\n', flush: bool = False, pre: any = '', post: any = '',
             send: bool = True) -> None:
        """
        Logs info.
        :param message: Message to be logged.
        :param end: End text of message
        :param flush: Clear or not clear current line
        :param pre: Text to add before string
        :param post: Text to add after the log name
        :param send: Determines whether to log the message.
        """

        self.__log(message, 'INFO', end=end, flush=flush, pre=pre, post=post, send=send)

    def err(self, message: any, end: str = '\n', flush: bool = False, pre: str = '', post: any = '',
            send: bool = True) -> None:
        """
        Logs errors.
        :param message: Message to be logged.
        :param end: End text of message
        :param flush: Clear or not clear current line
        :param pre: Text to add before string
        :param post: Text to add after the log name
        :param send: Determines whether to log the message.
        """

        self.__log(message, 'ERR', end=end, flush=flush, pre=pre, post=post, send=send)

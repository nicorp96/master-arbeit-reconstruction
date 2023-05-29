import tkinter
import logging
import datetime

# this item "module_logger" is visible only in this module,
# (but you can create references to the same logger object from other modules 
# by calling getLogger with an argument equal to the name of this module)
# this way, you can share or isolate loggers as desired across modules and across threads
# ...so it is module-level logging and it takes the name of this module (by using __name__)
# recommended per https://docs.python.org/2/library/logging.html
module_logger = logging.getLogger(__name__)

class simpleapp_tk(tkinter.Tk):
    def __init__(self,parent):
        tkinter.Tk.__init__(self,parent)
        self.parent = parent

        self.grid()

        self.mybutton = tkinter.Button(self, text="ClickMe")
        self.mybutton.grid(column=0,row=0,sticky='EW')
        self.mybutton.bind("<ButtonRelease-1>", self.button_callback)

        self.mytext = tkinter.Text(self, state="disabled")
        self.mytext.grid(column=0, row=1)

    def button_callback(self, event):
        now = datetime.datetime.now()
        module_logger.info(now)

class MyHandlerText(logging.StreamHandler):
    def __init__(self, textctrl):
        logging.StreamHandler.__init__(self) # initialize parent
        self.textctrl = textctrl

    def emit(self, record):
        msg = self.format(record)
        self.textctrl.config(state="normal")
        self.textctrl.insert("end", msg + "\n")
        self.flush()
        self.textctrl.config(state="disabled")

class GUI(tkinter.Tk):
    def __init__(self,parent):
        tkinter.Tk.__init__(self,parent)
        self._root = parent
        self._set_tq_buttons()

    def _set_tq_buttons(self):
        tkinter.Button(self, text="Quit", command=self.quit,
                  width=90).grid()
    
    def create_new_button(self, command, text ='new'):
        tkinter.Button(self, text=text, command=command,
                  width=90).grid()

    def text(self):
        text = tkinter.Text(self)
        text.pack()

    def start_main_loop(self):
        self.mainloop()

if __name__ == "__main__":

    # create Tk object instance
    app = simpleapp_tk(None)
    app.title('my application')

    # setup logging handlers using the Tk instance created above
    # the pattern below can be used in other threads...
    # ...to allow other thread to send msgs to the gui
    # in this example, we set up two handlers just for demonstration (you could add a fileHandler, etc)
    stderrHandler = logging.StreamHandler()  # no arguments => stderr
    module_logger.addHandler(stderrHandler)
    guiHandler = MyHandlerText(app.mytext)
    module_logger.addHandler(guiHandler)
    module_logger.setLevel(logging.INFO)
    module_logger.info("from main")

    # start Tk
    app.mainloop()

    gui = GUI(None)
    gui.start_main_loop()
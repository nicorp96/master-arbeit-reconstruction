import tkinter as tk
import enum


class MSG_TYPE(enum.Enum):
    USER = "USER:    "
    INFO = "INFO:     "
    ERROR = "ERROR: "


class GUI(tk.Tk):
    MAIN_TEXT = "THIS LABEL DISPLAYS THE INFOS AND USER TASKS FOR A SCANNING PROCESS"

    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self._root = parent
        self._label = tk.Label(
            self,
            text="",
            font=("Helvetica", 14),
            compound="left",
            background="white",
            justify=tk.LEFT,
            anchor="w",
            relief="sunken",
        )
        self._text_label = ""

    def _set_tq_buttons(self):
        bt = tk.Button(self, text="Quit", command=self.quit, width=90)
        bt.pack()

    def create_new_button(self, command, text="new", side="top"):
        button = tk.Button(self, text=text, command=command, width=90)
        button.pack(side=side)

    def text(self):
        text = tk.Text(self)
        text.pack()

    # todo: better names
    def label(self, text: str, type=MSG_TYPE.INFO):
        text_with_type = type.value + text
        self._text_label = self._text_label + "\n" + text_with_type
        self._label.configure(text=self._text_label)
        self._label.pack(fill="both")

    def remove_text(self):
        self._text_label = ""

    def create_gui_modality_1(
        self,
        initialize_process,
        first_process,
        turn_table_step_angle_process,
        create_and_save_mesh,
        visualize_main_object,
        color_optimization,
        restart,
    ):
        self.create_new_button(text="Initialize Scan", command=initialize_process)
        self.create_new_button(text="Start First Scann", command=first_process)
        self.create_new_button(
            text="Next Position / Angle", command=turn_table_step_angle_process
        )
        self.create_new_button(
            text="Create and Save Mesh", command=create_and_save_mesh
        )
        self.label(text=self.MAIN_TEXT, type=MSG_TYPE.INFO)
        self.create_new_button(
            text="Visualize and Save",
            command=visualize_main_object,
            side="left",
        )
        self.create_new_button(
            text="Color Optimization", command=color_optimization, side="right"
        )
        self.create_new_button(text="Restart Process", command=restart, side="right")
        self.mainloop()

    def create_gui_modality_2(
        self,
        initialize_process,
        scanning_process,
        create_and_save_mesh,
        create_and_save_textured_mesh,
        visualize_main_object,
        color_optimization,
        visualize_mesh,
    ):
        self.create_new_button(text="Initialize Scan", command=initialize_process)
        self.create_new_button(text="Scanning Process", command=scanning_process)
        self.create_new_button(text="Reconstruct Object PCL", command=create_and_save_mesh)
        self.create_new_button(
            text="Create textured Mesh", command=create_and_save_textured_mesh
        )
        self.label(text=self.MAIN_TEXT, type=MSG_TYPE.INFO)
        self.create_new_button(
            text="Visualize Mesh",
            command=visualize_mesh,
            side="left",
        )
        self.create_new_button(
            text="Visualize PCL", command=visualize_main_object, side="right"
        )
        self.create_new_button(
            text="Color Optimization", command=color_optimization, side="right"
        )
        self.mainloop()


# Work in Progress
class GUItk(tk.Tk):
    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self.parent = parent

        self.grid()

        self.quit_button = tk.Button(self.parent, text="Quit")
        self.init_button = tk.Button(self.parent, text="Initialize Scan")

        self.mytext = tk.Text(self, state="disabled")
        self.mytext.grid(column=0, row=1)

    def create_gui(self, func_init):
        self.quit_button.grid(column=0, row=0, sticky="EW")
        self.quit_button.bind("<ButtonRelease-1>", self.quit)
        self.init_button.grid(column=0, row=0, sticky="EW")
        self.init_button.bind("<ButtonRelease-1>", func_init)

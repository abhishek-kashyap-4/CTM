# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:17:15 2024

@author: kashy
"""

import tkinter as tk

def print_to_window(window, message):
    text_widget = tk.Text(window)
    text_widget.pack()
    text_widget.insert(tk.END, message)

root = tk.Tk()

if __name__ == '__main__':
    # Create main application window
    root = tk.Tk()
    root.title("Multiple Windows Example")
    
    # Create two additional windows
    window1 = tk.Toplevel(root)
    window1.title("Window 1")
    print_to_window(window1, 'apl')
    
    
    window1 = tk.Toplevel(root)
    window1.title("Window 2")
    print_to_window(window1, "This is window 2.")
    
    root.mainloop()
    

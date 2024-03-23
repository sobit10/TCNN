import PySimpleGUI as sg


def popup(full, pre):
    # layout = [
    #     [sg.Button(f'{" " * 7}Full-Analysis+Plots{" " * 7}'), sg.Button(f'{" " * 3}Plot Pre-Evaluated Data{" " * 3}'),
    #      sg.Button(f'{" " * 3}GUI Data{" " * 3}')]]
    layout = [
        [sg.Button(f'{" " * 7}Full-Analysis+Plots{" " * 7}'), sg.Button(f'{" " * 3}Plot Pre-Evaluated Data{" " * 3}')]]

    event, values = sg.Window('', layout).read(close=True)
    if event == f'{" " * 7}Full-Analysis+Plots{" " * 7}':
        full()
    elif event == f'{" " * 3}Plot Pre-Evaluated Data{" " * 3}':
        pre()
    # elif event == f'{" " * 3}GUI Data{" " * 3}':
    #     layout = [[sg.Button(f'{" " * 7}Dataset1{" " * 7}'), sg.Button(f'{" " * 7}Dataset2{" " * 7}'), sg.Button(f'{" " * 7}Dataset3{" " * 7}')]]
    #     event, values = sg.Window('', layout).read(close=True)
    #     if event == f'{" " * 7}Dataset1{" " * 7}':
    #         layout = [[sg.Text("Enter num (0<=398")],
    #                   [sg.Input()],
    #                   [sg.Button('Ok')]]
    #         window = sg.Window('select row', layout)
    #         event1, values1 = window.read()
    #         window.close()
    #         gui(int(event[-8])-1, int(values1[0]))
    #
    #     elif event == f'{" " * 7}Dataset2{" " * 7}':
    #         layout = [[sg.Text("Enter num (0<=31")],
    #                   [sg.Input()],
    #                   [sg.Button('Ok')]]
    #         window = sg.Window('select row', layout)
    #         event1, values1 = window.read()
    #         window.close()
    #         gui(int(event[-8])-1, int(values1[0]))
    #     elif event == f'{" " * 7}Dataset3{" " * 7}':
    #         layout = [[sg.Text("Enter num (0<=302")],
    #                   [sg.Input()],
    #                   [sg.Button('Ok')]]
    #         window = sg.Window('select row', layout)
    #         event1, values1 = window.read()
    #         window.close()
    #         gui(int(event[-8])-1, int(values1[0]))


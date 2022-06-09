class HTMLFile:
    def __init__(self):
        self.head = """<!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Title</title>
        <style>
            .box {
                width: 100%;
                background: #000;
            }
    
            .box tr th {
                background: #c4d4f5;
            }
    
            .box tr td {
                background: #fff;
                text-align: center;
            }
    
            .box tr:nth-of-type(2n+3) td {
                background: #fafafa;
            }
        </style>
    </head>
    <body>
    <table class='box' cellspacing='1px'>
        <tr>
            <th></th>
            <th>X(mm)</th>
            <th>Y(mm)</th>
            <th>Z(mm)</th>
        </tr>"""

        self.line = """    
        <tr>
            <td>{0}</td>
            <td>{1}</td>
            <td>{2}</td>
            <td>{3}</td>
        </tr>"""

        self.end = """
            </table>
            </body>
            </html>"""


def editHtmlFile(x, y, z):
    hfile = HTMLFile()
    file = hfile.head
    for i in range(len(x)):
        new_line = hfile.line.format(i + 1, round(x[i], 2), round(y[i], 2), round(z[i], 2))
        file = file + new_line
    file = file + hfile.end

    with open('./src/textBrowser_coor.html', 'w') as f:
        f.write(file)

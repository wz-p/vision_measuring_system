
filename = 'textBrowser.html'
filename_new = 'textBrowser_new.html'

fin = open('textBrowser.html', 'r')
fout = open('textBrowser_new.html', 'a')


a = 2
b = 3
c = 5
d = 7
for line in fin:
    if line.strip() == '<tr>':
      fout.write('\t\t<td>{0}</td>\n<td>{1}</td>\n<td>{2}</td>\n<td>{3}</td>'.format(a,b,c,d))
    fout.write(line)

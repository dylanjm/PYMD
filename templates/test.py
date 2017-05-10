import smtplib
 
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login("pymd.investments@gmail.com", "Pymdshizzam1234")
 
msg = "YOUR MESSAGE!"
server.sendmail("pymd.investments@gmail.com", "", msg)
server.quit()

from flask import Flask
from flask_mail import Mail, Message, Attachment

app = Flask(__name__)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'eliar.arge@gmail.com'
app.config['MAIL_PASSWORD'] = 'ximjomxcivdluwkv'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)

def send_mail(email, grafik_path):
    with app.app_context():
        msg = Message('Mind4Machines - TEXTRACK | YAbby Batatrya Tahmini', 
                      sender='eliar.arge@gmail.com',
                      recipients=['sencer.sultanoglu@eliarge.com','efe.demir@eliarge.com',' arge@mayteks.com'])#'sencer.sultanoglu@eliarge.com',' arge@mayteks.com',

        msg.body = "Batarya Omrunun Az kaldigi Tespit Edilen Yabbyler Verilmistir"

        # Open the file in bynary mode
        with open(grafik_path, 'rb') as f:
            img_data = f.read()

        # Attach the file
        msg.attach(filename='yabby_predictions.png', content_type='image/png', data=img_data)
        
        msg.html = "<h1>{}</h1>".format(email)
        mail.send(msg)

# def send_mail(email, grafik_path):
#     with app.app_context():
#         msg = Message('Mind4Machines - TEXTRACK | YAbby Batatrya Tahmini', 
#                     sender='eliar.arge@gmail.com',
#                     recipients=['efe.demir@eliarge.com'])#'sencer.sultanoglu@eliarge.com',' arge@mayteks.com',

#         msg.body = "Batarya Omrunun Az kaldigi Tespit Edilen Yabbyler Verilmistir"

#         # Open the file in bynary mode
#         with open(grafik_path, 'rb') as f:
#             img_data = f.read()

#         # Create an Attachment object
#         # You can change the content type as needed
#         attach_image = Attachment(filename=grafik_path, 
#                                   content_type='image/png', 
#                                   data=img_data)
#         msg.html = "<h1>{}</h1>".format(email)
#         # Add the attachment to your Message
#         print('Content type:', attach_image.content_type)

#         msg.attach(attach_image)
#         mail.send(msg)
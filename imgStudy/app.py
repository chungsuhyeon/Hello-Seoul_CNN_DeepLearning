from img_test import check_photo_str as ch
from flask import Flask
from flask_restx import Api, Resource
app = Flask(__name__)
api = Api(app)


@api.route('/hello/<string:imgname>')
class HelloWorld(Resource):
    def get(self,imgname):
        #img_test.check_photo_str("./img/searchImg/%s" .format(imgname))
        #DB connect

        #/web/resources/upload/deeplearning/filename
        #ai cnn learning
        #return = json
        #json{filename : imgname, food : foodname, accuracy : x%}

        # print("C:/Python_code/imgStudy/img/searchImg/"+imgname)

        return ch("C:/Python_code/imgStudy/img/searchImg/" + imgname,imgname)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)

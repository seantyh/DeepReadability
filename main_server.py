import cherrypy
from cherrypy import request, response
import decoder
import os

BASEPATH = os.path.dirname(__file__)


class ReadDecoder:

    def __init__(self):
        self.dec = decoder.Decoder(os.path.join(BASEPATH, "model.h5"))

    @cherrypy.expose
    def index(self):
        return 'Hello, World tre!'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def readability(self):
        if request.method != "POST":             
            return {
                "status": "error", 
                "msg": "Method not allowed"}

        intext = request.body.read().decode("UTF-8")
        if len(intext) == 0:
            return {
                "status": "error",
                "msg": "text is empty"
            }

        grade = self.dec.predict_text(intext)
        return {
                "status": "success",
                "msg": str(grade),
                "data": {
                    "grade": int(grade),
                    "stroke": self.dec.get_stroke_vec(intext),
                    "freqs": self.dec.get_freq_vec(intext),
                    "nchar": self.dec.get_char_count(intext)
                }
            }
    

cherrypy.config.update("server.conf")        
cherrypy.quickstart(ReadDecoder())

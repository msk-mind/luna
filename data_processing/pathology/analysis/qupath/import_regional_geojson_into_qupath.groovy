import qupath.lib.objects.*
import qupath.lib.roi.*
import com.google.gson.Gson
import java.io.FileReader;
import groovy.io.FileType
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Collectors;
import qupath.lib.geom.Point2



def imageData = getCurrentImageData()
def filename = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def slide_identifier = filename.replace(".svs", "").toString()    

// make sure to change URL, details can be found on confluence on how to configure the API's url accordingly.
// API FORMAT  http://{SERVER}:{PORT}/mind/api/v1/getPathologyAnnotation/{DMT_name}/{slide_identifier: either HobI-ID or Slide-ID}/regional/{labelset_name}
def url = "http://SERVER:PORT/mind/api/v1/getPathologyAnnotation/msk_mind_ov/" + slide_identifier + "/regional" + "/simplified_pixel_classifier_labels"
print(url)

def get = new URL(url).openConnection();
def getRC = get.getResponseCode();

if(getRC.equals(200)) {

    
    def hierarchy = imageData.getHierarchy()
    def text = get.getInputStream().getText();

    //Read into a map    
    def map = new Gson().fromJson(text, Map)
    
    
    annotations = []

    for (feat in map['features']) {
        def name = feat['properties']['label_name'].toString()
        def vertices = feat['geometry']['coordinates']
        def points = vertices.collect {new Point2(it[0], it[1])}
        def polygon = new PolygonROI(points)
        def pathAnnotation = new PathAnnotationObject(polygon)
        pathAnnotation.setPathClass(getPathClass(name))
        annotations << pathAnnotation
    }
    
    hierarchy.addPathObjects(annotations)
    println(hierarchy.getAnnotationObjects().size() + " annotations added to" + filename)
    

}


fireHierarchyUpdate()




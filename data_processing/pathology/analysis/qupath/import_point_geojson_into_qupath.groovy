import qupath.lib.io.GsonTools
import static qupath.lib.gui.scripting.QPEx.*

// Name of the subdirectory containing the TMA grid
def subDirectory = "Manual points"

// If true, don't check for existing points
boolean ignoreExisting = false

// Check we have an image open from a project
def hierarchy = getCurrentHierarchy()
if (hierarchy == null) {
    println "No image is available!"
    return
}
def name = getProjectEntry()?.getImageName()
if (name == null) {
    println "No image name found! Be sure to run this script with a project and image open."
    return    
}
    
// Resist adding (potentially duplicate) points unless the user explicitly requests this
def existingPoints = getAnnotationObjects().findAll {it.getROI()?.isPoint()}
if (!ignoreExisting && !existingPoints.isEmpty()) {
    println "Point annotations are already present! Please delete these first, or set ignoreExisting = true at the start of this script."
    return
}

def imageData = getCurrentImageData()
def filename = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

def image_id = filename.replace(".svs", "").toString()    
// make sure to change URL, details can be found on confluence on how to configure the API's url accordingly.
// API FORMAT  http://{SERVER}:{PORT}/mind/api/v1/getPathologyAnnotation/{PROJECT}/{image_id}/point/{labelset_name}
def url = "http://SERVER:PORT/mind/api/v1/getPathologyAnnotation/OV_16-158/" + image_id + "/point/lymphocyte_detection_labelset"
print(url)

def get = new URL(url).openConnection();
def getRC = get.getResponseCode();

if(getRC.equals(200)) {
    
    def text = get.getInputStream().getText();


    def type = new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>() {}.getType()
    def points = GsonTools.getInstance().fromJson(text, type)
    hierarchy.insertPathObjects(points)
    println(hierarchy.getAnnotationObjects().size() + " point annotations added to" + filename)

}

fireHierarchyUpdate()


import qupath.ext.stardist.StarDist2D
import org.slf4j.LoggerFactory;

def logger = LoggerFactory.getLogger("stardist_simple");

setImageType('BRIGHTFIELD_H_DAB');
def imageData = getCurrentImageData()
def server = getCurrentImageData().getServer()

def stains = imageData.getColorDeconvolutionStains()

logger.info ("Stains=${stains}")

// Specify the model file (you will need to change this!)
def pathModel = '/models/stardist/dsb2018_heavy_augment.pb'

var stardist = StarDist2D.builder(pathModel)
        .preprocess( // Extra preprocessing steps, applied sequentially
            ImageOps.Channels.deconvolve(stains),
            ImageOps.Channels.extract(0),
         )
        .pixelSize(0.5)
        .includeProbability(true)
        .threshold(0.25)
        .measureShape()
        .cellExpansion(8)
        .measureIntensity()
        .build()

// get dimensions of slide
maxX = server.getWidth()
maxY = server.getHeight()

// create rectangle roi (over entire area of image) for detections to be run over
def plane = ImagePlane.getPlane(0, 0)
def roi = ROIs.createRectangleROI(maxX * .1, maxY * .1, maxX * 0.8, maxY * 0.8, plane)
def annotationROI = PathObjects.createAnnotationObject(roi)
addObject(annotationROI)
selectAnnotations();

def pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}

logger.info("Running detection...")
stardist.detectObjects(imageData, pathObjects)

logger.info("Filtering detections Nucleus: Area µm^2 <= 10 or >= 200 ...")
def toDelete_small = getDetectionObjects().findAll {measurement(it, "Nucleus: Area µm^2") <= 10}
removeObjects(toDelete_small, true)

def toDelete_large = getDetectionObjects().findAll {measurement(it, "Nucleus: Area µm^2") >= 200}
removeObjects(toDelete_large, true)

logger.info("Started writing cell object data...")
saveDetectionMeasurements('/output_dir/cell_detections.tsv')

logger.info("Started writing cell object geojson...")
def detection_objects = getDetectionObjects()
def detection_geojson = GsonTools.getInstance(true)
new File('/output_dir/cell_detections.geojson').withWriter('UTF-8') {
    detection_geojson.toJson(detection_objects, it)
}

logger.info ("Done")
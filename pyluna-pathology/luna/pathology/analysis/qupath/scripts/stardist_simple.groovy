import qupath.ext.stardist.StarDist2D

setImageType('BRIGHTFIELD_H_DAB');

def imageData = getCurrentImageData()

def stains = imageData.getColorDeconvolutionStains()

println (stains)

// Specify the model file (you will need to change this!)
def pathModel = '/models/stardist/dsb2018_heavy_augment.pb'

var stardist = StarDist2D.builder(pathModel)
        .preprocess( // Extra preprocessing steps, applied sequentially
            ImageOps.Channels.deconvolve(stains),
            ImageOps.Channels.extract(0),
         )
        .pixelSize(0.5)
        .includeProbability(true)
        .threshold(0.5)
        .build()


def server = getCurrentImageData().getServer()

// get dimensions of slide
minX = 0
minY = 0
maxX = server.getWidth()
maxY = server.getHeight()

// create rectangle roi (over entire area of image) for detections to be run over
def plane = ImagePlane.getPlane(0, 0)
def roi = ROIs.createRectangleROI(maxX * .35, maxY * .25, maxX * 0.2, maxY * 0.2, plane)
def annotationROI = PathObjects.createAnnotationObject(roi)
addObject(annotationROI)
selectAnnotations();

def pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}

// Write the region of the image corresponding to the currently-selected object
def requestROI_pre = RegionRequest.createInstance(server.getPath(), 1.0, roi)
writeImageRegion(server, requestROI_pre, '/output_dir/original.tif')

println("Running!")

stardist.detectObjects(imageData, pathObjects)

imageData = getCurrentImageData()

def cellLabelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .useCells()
    .useUniqueLabels()
    .downsample(1.0)    // Choose server resolution; this should match the resolution at which tiles are exported    
    .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
    .build()

// write cell object geojsson (with tissue-type labels)
println("started writing cell object geojson")
boolean detection_pretty_print = true
def detections = getDetectionObjects()
def detection_gson = GsonTools.getInstance(detection_pretty_print)
new File('/output_dir/object_detection_results.geojson').withWriter('UTF-8') {
    detection_gson.toJson(detections, it)
}

// def viewer = getCurrentImageData()
// writeRenderedImage(viewer, '/output_dir/cell_detections.tif')
writeImageRegion(cellLabelServer, requestROI_pre, '/output_dir/cell_markup.tif')

// def requestROI_post = RegionRequest.createInstance(server.getPath(), 1.0, roi)
// writeImageRegion(server, requestROI_post, '/output_dir/cell_detections.tif')
saveDetectionMeasurements('/output_dir/cell_detections.csv')

def annotations = getAnnotationObjects()

// 'FEATURE_COLLECTION' is standard GeoJSON format for multiple objects
exportObjectsToGeoJson(annotations, '/output_dir/cell_detections.geojson', "FEATURE_COLLECTION")

println ("Done")
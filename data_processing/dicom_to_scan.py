"""
This script reads a folder of dicoms (passed as arguement) at DIR/inputs, and calls ITK methods to generate a MHD (scan) and associated ZRAW at DIR/outputs 
"""
import os
import sys
import itk

if len(sys.argv) < 4:
    print("Usage: " + sys.argv[0] +
          " [ProjectDirectory [DicomDirectory [outputFileExt]]]")

# program args
project_dir = sys.argv[1]
input_dir = sys.argv[2]
file_ext = sys.argv[3]

output_dir = os.path.join(project_dir, "scans")

PixelType = itk.ctype('signed short')
Dimension = 3

ImageType = itk.Image[PixelType, Dimension]

namesGenerator = itk.GDCMSeriesFileNames.New()
namesGenerator.SetUseSeriesDetails(True)
namesGenerator.AddSeriesRestriction("0008|0021")
namesGenerator.SetGlobalWarningDisplay(False)
namesGenerator.SetDirectory(input_dir)

seriesUIDs = namesGenerator.GetSeriesUIDs()
num_dicoms = len(seriesUIDs)

if num_dicoms < 1:
    print('No DICOMs in: ' + input_dir)
    sys.exit(1)

print('The directory {} contains {} DICOM Series: '.format(input_dir, str(num_dicoms)))
for uid in seriesUIDs:
    print(uid)

for uid in seriesUIDs:
    print('Reading: ' + uid)
    fileNames = namesGenerator.GetFileNames(uid)
    if len(fileNames) < 1: continue

    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.ForceOrthogonalDirectionOff()

    writer = itk.ImageFileWriter[ImageType].New()

    outFileName = os.path.join(output_dir, uid + '.' + file_ext)
    writer.SetFileName(outFileName)
    writer.UseCompressionOn()
    writer.SetInput(reader.GetOutput())
    print('Writing: ' + outFileName)
    writer.Update()

# Output filepath without extension
sys.stdout.write(os.path.join(output_dir, uid))

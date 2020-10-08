"""
This script reads a folder of dicoms (passed as arguement) at DIR/inputs, and calls ITK methods to generate a MHD (scan) and associated ZRAW at DIR/outputs 
"""
import os
import sys
import itk

if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] +
          " [DicomDirectory [outputFileExt [seriesName]]]")
    print("If DicomDirectory is not specified, current directory is used\n")

# current directory by default
dicom_dir = sys.argv[1]

input_dir = dicom_dir + "/inputs"
output_dir = dicom_dir + "/outputs"

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

outFileExt = 'mhd'
if len(sys.argv) > 2:
    outFileExt = sys.argv[2]

seriesFound = False
for uid in seriesUIDs:
    seriesIdentifier = uid
    if len(sys.argv) > 3:
        seriesIdentifier = sys.argv[3]
        seriesFound = True
    print('Reading: ' + seriesIdentifier)
    fileNames = namesGenerator.GetFileNames(seriesIdentifier)
    if len(fileNames) < 1: continue

    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.ForceOrthogonalDirectionOff()

    writer = itk.ImageFileWriter[ImageType].New()

    outFileName = os.path.join(output_dir, seriesIdentifier + '.' + outFileExt)
    writer.SetFileName(outFileName)
    writer.UseCompressionOn()
    writer.SetInput(reader.GetOutput())
    print('Writing: ' + outFileName)
    writer.Update()

    if seriesFound:
        break

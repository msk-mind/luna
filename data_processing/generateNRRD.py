#!/usr/bin/env python
import os
import sys
import itk


if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] +
          " [DicomDirectory [outputFileName [seriesName]]]")
    print("If DicomDirectory is not specified, current directory is used\n")

# current directory by default
dirName = '.'
if len(sys.argv) > 1:
    dirName = sys.argv[1]

PixelType = itk.ctype('signed short')
Dimension = 3

ImageType = itk.Image[PixelType, Dimension]

namesGenerator = itk.GDCMSeriesFileNames.New()
namesGenerator.SetUseSeriesDetails(True)
namesGenerator.AddSeriesRestriction("0008|0021")
namesGenerator.SetGlobalWarningDisplay(False)
namesGenerator.SetDirectory(dirName)

seriesUID = namesGenerator.GetSeriesUIDs()

if len(seriesUID) < 1:
    print('No DICOMs in: ' + dirName)
    sys.exit(1)

print('The directory: ' + dirName)
print('Contains the following DICOM Series: ')
for uid in seriesUID:
    print(uid)

seriesFound = False
for uid in seriesUID:
    seriesIdentifier = uid
    if len(sys.argv) > 3:
        seriesIdentifier = sys.argv[3]
        seriesFound = True
    print('Reading: ' + seriesIdentifier)
    fileNames = namesGenerator.GetFileNames(seriesIdentifier)

    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.ForceOrthogonalDirectionOff()

    writer = itk.ImageFileWriter[ImageType].New()
    outFileExt = 'nrrd'
    if len(sys.argv) > 2:
        outFileExt = sys.argv[2]
    outFileName = os.path.join(dirName, "outputs", seriesIdentifier + '.' + outFileExt)
    writer.SetFileName(outFileName)
    writer.UseCompressionOn()
    writer.SetInput(reader.GetOutput())
    print('Writing: ' + outFileName)
    writer.Update()

    if seriesFound:
        break


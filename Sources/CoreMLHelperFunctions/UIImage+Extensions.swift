/*
  Copyright (c) 2017-2021 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

#if canImport(UIKit)

import UIKit
import CoreML

extension UIImage {
  /**
    Resizes the image.

    - Parameter scale: If this is 1, `newSize` is the size in pixels.
  */
  @nonobjc public func resized(to newSize: CGSize, scale: CGFloat = 1) -> UIImage {
    let format = UIGraphicsImageRendererFormat.default()
    format.scale = scale
    let renderer = UIGraphicsImageRenderer(size: newSize, format: format)
    let image = renderer.image { _ in
      draw(in: CGRect(origin: .zero, size: newSize))
    }
    return image
  }
    
    /**
      Resizes the image with cgImage.

      - Parameter scale: If this is 1, `newSize` is the size in pixels.
    */
    public func resizedImage(for size: CGSize) -> UIImage? {
            let image = self.cgImage
            let context = CGContext(data: nil,
                                    width: Int(size.width),
                                    height: Int(size.height),
                                    bitsPerComponent: image!.bitsPerComponent,
                                    bytesPerRow: Int(size.width),
                                    space: image?.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!,
                                    bitmapInfo: image!.bitmapInfo.rawValue)
            context?.interpolationQuality = .high
            context?.draw(image!, in: CGRect(origin: .zero, size: size))

            guard let scaledImage = context?.makeImage() else { return nil }

            return UIImage(cgImage: scaledImage)
        }

  /**
    Rotates the image around its center.

    - Parameter degrees: Rotation angle in degrees.
    - Parameter keepSize: If true, the new image has the size of the original
      image, so portions may be cropped off. If false, the new image expands
      to fit all the pixels.
  */
  @nonobjc public func rotated(by degrees: CGFloat, keepSize: Bool = true) -> UIImage {
    let radians = degrees * .pi / 180
    let newRect = CGRect(origin: .zero, size: size).applying(CGAffineTransform(rotationAngle: radians))

    // Trim off the extremely small float value to prevent Core Graphics from rounding it up.
    var newSize = keepSize ? size : newRect.size
    newSize.width = floor(newSize.width)
    newSize.height = floor(newSize.height)

    return UIGraphicsImageRenderer(size: newSize).image { rendererContext in
      let context = rendererContext.cgContext
      context.setFillColor(UIColor.black.cgColor)
      context.fill(CGRect(origin: .zero, size: newSize))
      context.translateBy(x: newSize.width / 2, y: newSize.height / 2)
      context.rotate(by: radians)
      let origin = CGPoint(x: -size.width / 2, y: -size.height / 2)
      draw(in: CGRect(origin: origin, size: size))
    }
  }
    
    class func imageFromColor(color: UIColor, size: CGSize=CGSize(width: 1, height: 1), scale: CGFloat) -> UIImage? {
       UIGraphicsBeginImageContextWithOptions(size, false, scale)
       color.setFill()
       UIRectFill(CGRect(origin: CGPoint.zero, size: size))
       let image = UIGraphicsGetImageFromCurrentImageContext()
       UIGraphicsEndImageContext()
       return image
   }
    
    public func mlMultiArray(scale preprocessScale:Double=1/255, rBias preprocessRBias:Double=0, gBias preprocessGBias:Double=0, bBias preprocessBBias:Double=0) -> MLMultiArray {
        let imagePixel = self.getPixelRgb(scale: preprocessScale, rBias: preprocessRBias, gBias: preprocessGBias, bBias: preprocessBBias)
//        let size = self.size
        let imagePointer : UnsafePointer<Double> = UnsafePointer(imagePixel)
        let mlArray = try! MLMultiArray(shape: [1,3,  NSNumber(value: Float(512)), NSNumber(value: Float(512))], dataType: MLMultiArrayDataType.double)
        mlArray.dataPointer.initializeMemory(as: Double.self, from: imagePointer, count: imagePixel.count)
    
        return mlArray
    }
    
    public func mlMultiArrayGrayScale(scale preprocessScale:Double=1/255,bias preprocessBias:Double=0) -> MLMultiArray {
        let imagePixel = self.getPixelGrayScale(scale: preprocessScale, bias: preprocessBias)
//        let size = self.size
        let imagePointer : UnsafePointer<Double> = UnsafePointer(imagePixel)
        let mlArray = try! MLMultiArray(shape: [1,1,  NSNumber(value: Float(512)), NSNumber(value: Float(512))], dataType: MLMultiArrayDataType.double)
        mlArray.dataPointer.initializeMemory(as: Double.self, from: imagePointer, count: imagePixel.count)
        return mlArray
    }
    
    public func mlMultiArrayComposite(outImage out:UIImage, inputImage input:UIImage, maskImage mask: UIImage, scale preprocessScale:Double=1/255, rBias preprocessRBias:Double=0, gBias preprocessGBias:Double=0, bBias preprocessBBias:Double=0) -> MLMultiArray {
        let imagePixel = self.getMaskedPixelRgb(out: out, input: input, mask: mask)
//        let size = self.size
        let imagePointer : UnsafePointer<Double> = UnsafePointer(imagePixel)
        let mlArray = try! MLMultiArray(shape: [1,3,  NSNumber(value: Float(512)), NSNumber(value: Float(512))], dataType: MLMultiArrayDataType.double)
        mlArray.dataPointer.initializeMemory(as: Double.self, from: imagePointer, count: imagePixel.count)
    
        return mlArray
    }
   
    public func getMaskedPixelRgb(out: UIImage,input: UIImage, mask:UIImage, scale preprocessScale:Double=1, rBias preprocessRBias:Double=0, gBias preprocessGBias:Double=0, bBias preprocessBBias:Double=0) -> [Double]
    {
        guard let outCGImage = out.cgImage?.resize(size: CGSize(width: 512, height: 512)) else {
            return []
        }
        let outbytesPerRow = outCGImage.bytesPerRow
        let outwidth = outCGImage.width
        let outheight = outCGImage.height
        let outbytesPerPixel = 4
        let outpixelData = outCGImage.dataProvider!.data! as Data

        guard let inputCGImage = input.cgImage?.resize(size: CGSize(width: 512, height: 512)) else {
            return []
        }
        let inputpixelData = inputCGImage.dataProvider!.data! as Data
        
        guard let maskCgImage = mask.cgImage?.resize(size: CGSize(width: 512, height: 512)) else {
            return []
        }
        let maskBytesPerRow = maskCgImage.bytesPerRow
        let maskBytesPerPixel = 4
        let maskPixelData = maskCgImage.dataProvider!.data! as Data

        var r_buf : [Double] = []
        var g_buf : [Double] = []
        var b_buf : [Double] = []

        for j in 0..<outheight {
            for i in 0..<outwidth {
                let pixelInfo = outbytesPerRow * j + i * outbytesPerPixel
                let maskPixelInfo = maskBytesPerRow * j + i * maskBytesPerPixel
                let v = Double(maskPixelData[maskPixelInfo])
                
                let r = Double(outpixelData[pixelInfo])
                let g = Double(outpixelData[pixelInfo+1])
                let b = Double(outpixelData[pixelInfo+2])
                let bgr = Double(inputpixelData[pixelInfo+1])
                let bgg = Double(inputpixelData[pixelInfo+2])
                let bgb = Double(inputpixelData[pixelInfo+3])
                if v > 0 {
                    r_buf.append(Double(r*preprocessScale)+preprocessRBias)
                    g_buf.append(Double(g*preprocessScale)+preprocessGBias)
                    b_buf.append(Double(b*preprocessScale)+preprocessBBias)
                } else {
                    r_buf.append(Double(bgr*preprocessScale)+preprocessRBias)
                    g_buf.append(Double(bgg*preprocessScale)+preprocessGBias)
                    b_buf.append(Double(bgb*preprocessScale)+preprocessBBias)

                }
            }
        }

        return ((r_buf + g_buf) + b_buf)
    }
    
    public func getPixelRgb(scale preprocessScale:Double=1/255, rBias preprocessRBias:Double=0, gBias preprocessGBias:Double=0, bBias preprocessBBias:Double=0) -> [Double]
    {
        guard let cgImage = self.cgImage?.resize(size: CGSize(width: 512, height: 512)) else {
            return []
        }
        let bytesPerRow = cgImage.bytesPerRow
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let pixelData = cgImage.dataProvider!.data! as Data

        var r_buf : [Double] = []
        var g_buf : [Double] = []
        var b_buf : [Double] = []
        
        for j in 0..<height {
            for i in 0..<width {
                let pixelInfo = bytesPerRow * j + i * bytesPerPixel
                let r = Double(pixelData[pixelInfo+1])
                let g = Double(pixelData[pixelInfo+2])
                let b = Double(pixelData[pixelInfo+3])
                r_buf.append(Double(r*preprocessScale)+preprocessRBias)
                g_buf.append(Double(g*preprocessScale)+preprocessGBias)
                b_buf.append(Double(b*preprocessScale)+preprocessBBias)
            }
        }

        return ((r_buf + g_buf) + b_buf)
    }
    
    public func getPixelGrayScale(scale preprocessScale:Double=1/255, bias preprocessBias:Double=0) -> [Double]
    {
        guard let cgImage = self.cgImage?.resize(size: CGSize(width: 512, height: 512)) else {
            return []
        }
        let bytesPerRow = cgImage.bytesPerRow
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let pixelData = cgImage.dataProvider!.data! as Data
        
        var buf : [Double] = []
        
        for j in 0..<height {
            for i in 0..<width {
                let pixelInfo = bytesPerRow * j + i * bytesPerPixel
                let v = Double(pixelData[pixelInfo])
                buf.append(Double(v*preprocessScale)+preprocessBias)
            }
        }
        return buf
    }
    
}

#endif


extension CGImage {
   public  func resize(size:CGSize) -> CGImage? {
        let width: Int = Int(size.width)
        let height: Int = Int(size.height)

        let bytesPerPixel = self.bitsPerPixel / self.bitsPerComponent
        let destBytesPerRow = width * bytesPerPixel


        guard let colorSpace = self.colorSpace else { return nil }
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: self.bitsPerComponent, bytesPerRow: destBytesPerRow, space: colorSpace, bitmapInfo: self.alphaInfo.rawValue) else { return nil }

        context.interpolationQuality = .high
        context.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))

        return context.makeImage()
    }
}

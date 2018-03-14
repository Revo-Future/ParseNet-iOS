import UIKit
import AVFoundation
import CoreVideo
import Metal


extension UIImage {
    func fixImageOrientation() -> UIImage? {
        
        guard let cgImage = self.cgImage else {
            return nil
        }
        
        if self.imageOrientation == UIImageOrientation.up {
            return self
        }
        
        let width  = self.size.width
        let height = self.size.height
        
        var transform = CGAffineTransform.identity
        
        switch self.imageOrientation {
        case .down, .downMirrored:
            transform = transform.translatedBy(x: width, y: height)
            transform = transform.rotated(by: CGFloat.pi)
            
        case .left, .leftMirrored:
            transform = transform.translatedBy(x: width, y: 0)
            transform = transform.rotated(by: 0.5*CGFloat.pi)
            
        case .right, .rightMirrored:
            transform = transform.translatedBy(x: 0, y: height)
            transform = transform.rotated(by: -0.5*CGFloat.pi)
            
        case .up, .upMirrored:
            break
        }
        
        switch self.imageOrientation {
        case .upMirrored, .downMirrored:
            transform = transform.translatedBy(x: width, y: 0)
            transform = transform.scaledBy(x: -1, y: 1)
            
        case .leftMirrored, .rightMirrored:
            transform = transform.translatedBy(x: height, y: 0)
            transform = transform.scaledBy(x: -1, y: 1)
            
        default:
            break;
        }
        
        // Now we draw the underlying CGImage into a new context, applying the transform
        // calculated above.
        guard let colorSpace = cgImage.colorSpace else {
            return nil
        }
        
        guard let context = CGContext(
            data: nil,
            width: Int(width),
            height: Int(height),
            bitsPerComponent: cgImage.bitsPerComponent,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: UInt32(cgImage.bitmapInfo.rawValue)
            ) else {
                return nil
        }
        
        context.concatenate(transform);
        
        switch self.imageOrientation {
            
        case .left, .leftMirrored, .right, .rightMirrored:
            // Grr...
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: height, height: width))
            
        default:
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        
        // And now we just create a new UIImage from the drawing context
        guard let newCGImg = context.makeImage() else {
            return nil
        }
        
        let img = UIImage(cgImage: newCGImg)
        
        return img;
    }
    
//    func fixImageOrientation() -> UIImage? {
//        var flip:Bool = false //used to see if the image is mirrored
//        var isRotatedBy90:Bool = false // used to check whether aspect ratio is to be changed or not
//        
//        var transform = CGAffineTransform.identity
//        
//        //check current orientation of original image
//        switch self.imageOrientation {
//        case .down, .downMirrored:
//            transform = transform.rotated(by: CGFloat(M_PI));
//            
//        case .left, .leftMirrored:
//            transform = transform.rotated(by: CGFloat(M_PI_2));
//            isRotatedBy90 = true
//        case .right, .rightMirrored:
//            //transform = transform.rotated(by: CGFloat(-M_PI_2));
//            transform = CGAffineTransform(rotationAngle: 0.5 * CGFloat.pi )
//
//            isRotatedBy90 = true
//            
//        case .up, .upMirrored:
//            break
//        }
//        
//        switch self.imageOrientation {
//            
//        case .upMirrored, .downMirrored:
//            transform = transform.translatedBy(x: self.size.width, y: 0)
//            flip = true
//            
//        case .leftMirrored, .rightMirrored:
//            transform = transform.translatedBy(x: self.size.height, y: 0)
//            flip = true
//        default:
//            break;
//        }
//        
//        // calculate the size of the rotated view's containing box for our drawing space
//        //let rotatedViewBox = UIView(frame: CGRect(origin: CGPoint(x:0, y:0), size: size))
//        let rotatedViewBox: UIView = UIView(frame: CGRect(x: 0, y: 0, width: self.size.height, height: self.size.width))
////        var rotatedViewBox: UIView = UIView(frame: CGRect(x: 0, y: 0, width: oldImage.size.width, height: oldImage.size.height))
//        
//        rotatedViewBox.transform = transform
//        let rotatedSize = rotatedViewBox.frame.size
//        
//        // Create the bitmap context
//        UIGraphicsBeginImageContext(rotatedSize)
//        let bitmap = UIGraphicsGetCurrentContext()
//        
//        // Move the origin to the middle of the image so we will rotate and scale around the center.
//        bitmap!.translateBy(x: rotatedSize.width / 2.0, y: rotatedSize.height / 2.0);
//        
//        bitmap?.rotate(by: 0.5 * CGFloat.pi)
//        
//        // Now, draw the rotated/scaled image into the context
//        var yFlip: CGFloat
//        
//        if(flip){
//            yFlip = CGFloat(-1.0)
//        } else {
//            yFlip = CGFloat(1.0)
//        }
//        
//        bitmap!.scaleBy(x: yFlip, y: -1.0)
//        
//        //check if we have to fix the aspect ratio
//        if isRotatedBy90 {
//            bitmap?.draw(self.cgImage!, in: CGRect(x: -size.height / 2, y: -size.width / 2, width: size.height,height: size.width))
//        } else {
//            bitmap?.draw(self.cgImage!, in: CGRect(x: -size.width / 2, y: -size.height / 2, width: size.width,height: size.height))
//        }
//        
//        let fixedImage = UIGraphicsGetImageFromCurrentImageContext()
//        UIGraphicsEndImageContext()
//        
//        print("before")
//        print(self.size)
//        print(self.size.width)
//        print(self.size.height)
//        print("after")
//        print(fixedImage?.size.width)
//        print(fixedImage?.size.height)
//        print(fixedImage?.imageOrientation.rawValue)
//        return fixedImage
//    }
}


func imageRotatedByDegrees(oldImage: UIImage, deg degrees: CGFloat) -> UIImage {
    //Calculate the size of the rotated view's containing box for our drawing space
    
    var rotatedViewBox: UIView = UIView(frame: CGRect(x: 0, y: 0, width: oldImage.size.width, height: oldImage.size.height))
    
    print(oldImage.size.height)
    if (oldImage.imageOrientation == .right)
    {
        rotatedViewBox = UIView(frame: CGRect(x: 0, y: 0, width: oldImage.size.height, height: oldImage.size.width))
    }
    
    let t: CGAffineTransform = CGAffineTransform(rotationAngle: degrees * CGFloat.pi / 180)
    rotatedViewBox.transform = t
    let rotatedSize: CGSize = rotatedViewBox.frame.size
    //Create the bitmap context
    UIGraphicsBeginImageContext(rotatedSize)
    let bitmap: CGContext = UIGraphicsGetCurrentContext()!
    //Move the origin to the middle of the image so we will rotate and scale around the center.
    bitmap.translateBy(x: rotatedSize.width / 2, y: rotatedSize.height / 2)
    //Rotate the image context
    bitmap.rotate(by: (degrees * CGFloat.pi / 180))
    //Now, draw the rotated/scaled image into the context
    bitmap.scaleBy(x: 1.0, y: -1.0)
    bitmap.draw(oldImage.cgImage!, in: CGRect(x: -oldImage.size.height / 2, y: -oldImage.size.width / 2, width: oldImage.size.height, height: oldImage.size.width))
    let newImage: UIImage = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()
    return newImage
}

public protocol VideoCaptureDelegate: class {
  func didCapture(texture: MTLTexture?, previewImage: UIImage?)
}

public class VideoCapture: NSObject, AVCapturePhotoCaptureDelegate {

  public var previewLayer: AVCaptureVideoPreviewLayer?
  public weak var delegate: VideoCaptureDelegate?

  var device: MTLDevice!
  var captureSession: AVCaptureSession!
	var photoOutput: AVCapturePhotoOutput!
  var textureCache: CVMetalTextureCache?

  public init(device: MTLDevice) {
    self.device = device
    super.init()
    setUp()
  }

  private func setUp() {
    guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache) == kCVReturnSuccess else {
      print("Error: Could not create a texture cache")
      return
    }

    captureSession = AVCaptureSession()
    captureSession.beginConfiguration()
    //captureSession.sessionPreset = AVCaptureSessionPresetMedium
    captureSession.sessionPreset = AVCaptureSessionPreset640x480

    guard let videoDevice = AVCaptureDevice.defaultDevice(withMediaType: AVMediaTypeVideo) else {
      print("Error: no video devices available")
      return
    }

    guard let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else {
      print("Error: could not create AVCaptureDeviceInput")
      return
    }

    if captureSession.canAddInput(videoInput) {
      captureSession.addInput(videoInput)
    }

    if let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession) {
      previewLayer.videoGravity = AVLayerVideoGravityResizeAspect
      previewLayer.connection?.videoOrientation = .portrait
      //  previewLayer.connection?.videoOrientation = .landscapeLeft
      self.previewLayer = previewLayer
    }

    photoOutput = AVCapturePhotoOutput()
    

    if captureSession.canAddOutput(photoOutput) {
      captureSession.addOutput(photoOutput)
    }
    captureSession.commitConfiguration()
  }

  public func start() {
    captureSession.startRunning()
  }

  /* Captures a single frame of the camera input. */
  public func captureFrame() {
    let settings = AVCapturePhotoSettings(format: [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ])

    settings.previewPhotoFormat = [
      kCVPixelBufferPixelFormatTypeKey as String: settings.availablePreviewPhotoPixelFormatTypes[0],
      kCVPixelBufferWidthKey as String: 640,
      kCVPixelBufferHeightKey as String: 480,
    ]
photoOutput.connection(withMediaType: AVMediaTypeVideo).videoOrientation = .portrait
    photoOutput?.capturePhoto(with: settings, delegate: self)
  }

  public func capture(_ captureOutput: AVCapturePhotoOutput,
                      didFinishProcessingPhotoSampleBuffer photoSampleBuffer: CMSampleBuffer?,
                      previewPhotoSampleBuffer: CMSampleBuffer?,
                      resolvedSettings: AVCaptureResolvedPhotoSettings,
                      bracketSettings: AVCaptureBracketedStillImageSettings?,
                      error: Error?) {

    var imageTexture: MTLTexture?
    var previewImage: UIImage?

    // Convert the photo to a Metal texture.
    if error == nil, let textureCache = textureCache,
       let sampleBuffer = photoSampleBuffer,
       let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

      let width = CVPixelBufferGetWidth(imageBuffer)
      let height = CVPixelBufferGetHeight(imageBuffer)

      var texture: CVMetalTexture?
      CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache,
          imageBuffer, nil, .bgra8Unorm, width, height, 0, &texture)

      if let texture = texture {
        imageTexture = CVMetalTextureGetTexture(texture)
      }
    }

    // Convert the preview to a UIImage and show it on the screen.
    if error == nil, let sampleBuffer = previewPhotoSampleBuffer,
       let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
//modify the orientation from imageBuffer here, if needed.
      let width = CVPixelBufferGetWidth(imageBuffer)
      let height = CVPixelBufferGetHeight(imageBuffer)
      let rect = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))

      let ciImage = CIImage(cvPixelBuffer: imageBuffer)
      let ciContext = CIContext(options: nil)
      if let cgImage = ciContext.createCGImage(ciImage, from: rect) {
        //previewImage = UIImage(cgImage: cgImage)
        //convert the orientation for display
        previewImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: UIImageOrientation.right)
        
        //previewImage = imageRotatedByDegrees(oldImage: previewImage!, deg: 270)
        //previewImage = previewImage?.fixImageOrientation()
      }
    }
    

    delegate?.didCapture(texture: imageTexture, previewImage: previewImage)
  }
}

//
//  ViewController.swift
//  EdgeNetsCV
//
//  Created by Sachin on 7/2/19.
//  Copyright © 2019 Sachin Mehta. All rights reserved.
//

import UIKit
import AVFoundation
import AVKit
import Vision
import CoreML
import Accelerate
import VideoToolbox


class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    @IBOutlet weak var cameraView: UIView!
    
    @IBOutlet weak var segView: UIImageView!
    
    private var requests = [VNRequest]()
    
    //espnet model
    let espnet_model = espnetv2_pascal_256()
    
    //new variables
    var bufferSize: CGSize = .zero
    var rootLayer: CALayer! = nil
    
    private let session = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let videoDataOutputQueue = DispatchQueue(label: "videoQueue", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    private var previewLayer: AVCaptureVideoPreviewLayer! = nil
    
    //define the filter that will convert the grayscale prediction to color image
    let masker = ColorMasker()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        setupAVCapture()
        
        //setup vision parts
        setupVisionModel()
        
        //start the capture
        startCaptureSession()
    }
    
    func startCaptureSession(){
        session.startRunning()
    }
    
    func setupAVCapture(){
        var deviceInput: AVCaptureDeviceInput!
        
        //select a video device
        let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera],
                                                           mediaType: .video,
                                                           position: .back).devices.first
        
        do {
            deviceInput = try AVCaptureDeviceInput(device: videoDevice!)
        } catch {
            print("Could not create video device input: \(error)")
            return
        }
        
        session.beginConfiguration()
        
        //video format
        session.sessionPreset = .vga640x480
        
        //add video input
        guard session.canAddInput(deviceInput) else{
            print("Could not add video device input to the session")
            session.commitConfiguration()
            return
        }
        
        session.addInput(deviceInput)
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            
            //add video data output
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        } else{
            print("Could not add video data output to the session")
            session.commitConfiguration()
            return
        }
        
        let captureConnection = videoDataOutput.connection(with: .video)
        //always process the frames
        captureConnection?.isEnabled = true
        do{
            try videoDevice!.lockForConfiguration()
            let dimensions = CMVideoFormatDescriptionGetDimensions((videoDevice?.activeFormat.formatDescription)!)
            bufferSize.width = CGFloat(dimensions.width)
            bufferSize.height = CGFloat(dimensions.height)
            videoDevice!.unlockForConfiguration()
        } catch{
            print(error)
        }
        
        session.commitConfiguration()
        
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        rootLayer = cameraView.layer
        rootLayer.bounds = cameraView.bounds
        previewLayer.frame = rootLayer.bounds
        rootLayer.addSublayer(previewLayer)
        
    }
    
    func setupVisionModel() {
        guard let visionModel = try? VNCoreMLModel(for: espnet_model.model) else{
            fatalError("Can not load CNN model")
        }
        
        let segmentationRequest = VNCoreMLRequest(model: visionModel, completionHandler: {request, error in
            DispatchQueue.main.async(execute: {
                if let results = request.results {
                    self.processSegmentationRequest(results)
                }
            })
        })
        segmentationRequest.imageCropAndScaleOption = .scaleFill
        self.requests = [segmentationRequest]
    }
    
    func processSegmentationRequest(_ observations: [Any]){
        let obs = observations as! [VNPixelBufferObservation]
        
        if obs.isEmpty{
            print("Empty")
        }
        
        let outPixelBuffer = (obs.first)!

        let segMaskGray = CIImage(cvPixelBuffer: outPixelBuffer.pixelBuffer)
        
        //pass through the filter that converts grayscale image to different shades of red
        self.masker.inputGrayImage = segMaskGray
        
        // add to the image view
        self.segView.image = UIImage(ciImage: self.masker.outputImage!, scale: 1.0, orientation: .right)
    }
    
    
    // this function notifies AVCatpreuDelegate everytime a new frame is received
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {return}
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])

        do {
            try imageRequestHandler.perform(self.requests)
        } catch{
            print(error)
        }
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
}


//converts the Grayscale image to RGB
// provides different shades of red based on pixel values
class ColorMasker: CIFilter
{
    var inputGrayImage : CIImage?
    
    let colormapKernel = CIColorKernel(source:
        "kernel vec4 colorMasker(__sample gray)" +
            "{" +
            " if (gray.r == 0.0f) {return vec4(0.0, 0.0, 0.0, 1.0);}" +
            "   return vec4(1.0, gray.r, gray.r, 1.0);" +
        "}"
    )
    
    override var attributes: [String : Any]
    {
        return [
            kCIAttributeFilterDisplayName: "Color masker",
            
            "inputGrayImage": [kCIAttributeIdentity: 0,
                              kCIAttributeClass: "CIImage",
                              kCIAttributeDisplayName: "Grayscale Image",
                              kCIAttributeType: kCIAttributeTypeImage
            ]
        ]
    }
    
    override var outputImage: CIImage!
    {
        guard let inputGrayImage = inputGrayImage,
            let colormapKernel = colormapKernel else
        {
            return nil
        }
        
        let extent = inputGrayImage.extent
        let arguments = [inputGrayImage]
        
        return colormapKernel.apply(extent: extent, arguments: arguments)
    }
}

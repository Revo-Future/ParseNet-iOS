//
//  Unpooling.swift
//  ParseNet
//
//  Created by Wenbo Huang on 2017/8/3.
//  Copyright © 2017年 Hollance. All rights reserved.
//

import Foundation

import Metal
import MetalPerformanceShaders
import Accelerate


/**
 unpooling layer
 
 
 */
public class Unpooling {
    let pipeline: MTLComputePipelineState
    
    
    public let device: MTLDevice
    
    public var offset = MPSOffset(x: 0, y: 0, z: 0)
    public var clipRect = MPSRectNoClip
    public var destinationFeatureChannelOffset = 0
    public var edgeMode = MPSImageEdgeMode.zero
    /**
     Creates a new unpooling output object.
     
     - Parameters:
     - featureChannels: the feature channel of input feature map.
     - width: the output feature map width.
     - height: the output feature map height.
     */
    public init(device: MTLDevice,
                featureChannels: Int,
                width: Int = 32,
                height: Int = 32 ) {
    
        let functionName: String
        if featureChannels <= 4 {
            functionName = "unpooling"
        } else {
            functionName = "unpooling_array"
        }
        
        
//        do {
//            let library = device.newDefaultLibrary()!
//            let unpool = library.makeFunction(name: functionName)
//            pipeline = try device.makeComputePipelineState(function: unpool!)
//            
//        } catch {
//            fatalError("Error initializing compute pipeline")
//        }

        pipeline = makeFunction(device: device, name: functionName, useForgeLibrary: false)
        
        self.device = device
    }
    
    public func encode(commandBuffer: MTLCommandBuffer,
                       sourceImage: MPSImage, destinationImage: MPSImage) {
        
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setTexture(destinationImage.texture, at: 1)
        encoder.dispatch(pipeline: pipeline, image: destinationImage)
        encoder.endEncoding()
        
        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
}

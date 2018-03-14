import MetalPerformanceShaders
import QuartzCore

private func makePool(device: MTLDevice) -> MPSCNNPoolingAverage {
    // only one average pooling layer in ParseNet, 32x32(256/8), stride 32.
    let pool = MPSCNNPoolingAverage(device: device,
                                    kernelWidth: 32,
                                    kernelHeight: 32,
                                    strideInPixelsX: 32,
                                    strideInPixelsY: 32)
    pool.offset = MPSOffset(x: 16, y: 16, z: 0)
    pool.edgeMode = MPSImageEdgeMode.clamp
    return pool
}

private func makeNormalize(device: MTLDevice) ->MPSCNNCrossChannelNormalization{
    let norm = MPSCNNCrossChannelNormalization(
        device: device,
        kernelSize: 2048)
    norm.alpha = 2048
    norm.beta = 0.5
    norm.delta = 0.0
    return norm

}


/*
 Implements the ParseNet, the basic network is MobileNet.
 */
public class ParseNet {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    // The custom compute kernels for preprocessing the input images.
    let pipelineRGB: MTLComputePipelineState
    let pipelineBGR: MTLComputePipelineState
    
    let outputImage: MPSImage
    let seg_img: MPSImage
    
    // The neural network expects a 256x256 pixel image. We use a lanczos filter
    // to scale the input image down to these dimensions.
    let lanczos: MPSImageLanczosScale
    let lanczos_large: MPSImageLanczosScale
    
    // After the last layer (fc7), we take the "softmax" of each output neuron.
    // This converts the last layer into a 1000-element vector of probabilities,
    // where each element in this vector corresponds to an ImageNet class label.
    let softmax: MPSCNNSoftMax
    
    /* The layers in the network: */
    let conv1_s2: MPSCNNConvolution  // 256x256x3  input, kernels (3x3x3x32  = 864 weights + 32 bias). s=2,p=1
    
    let conv2_1_dw: DepthwiseConvolutionKernel  // 128x128x32 input, kernels (3x3x32 = 288 weights + 32 bias) s=1,p=1
    let conv2_1_s1: MPSCNNConvolution  // 128x128x32 input, kernels (1x1x32x64 = 2048 weights + 64 bias) s=1,p=0
    let conv2_2_dw: DepthwiseConvolutionKernel // 128x128x64 input, kernels (3x3x64 = 576 weights + 64 bias) s=2,p=1
    let conv2_2_s1: MPSCNNConvolution // 64x64x64 input, kernels (1x1x64x128 = 8912 weights + 128 bias) s=1,p=0
    
    let conv3_1_dw: DepthwiseConvolutionKernel // 64x64x128 input, kernels (3x3x128 = 1152 weights + 128 bias) s=1,p=1
    let conv3_1_s1: MPSCNNConvolution // 64x64x128 input, kernels (1x1x128x128 = 16384 weights + 128 bias) s=1,p=0
    let conv3_2_dw: DepthwiseConvolutionKernel // 64x64x128 input, kernels (3x3x128 = 1152 weights + 128 bias) s=2,p=1
    let conv3_2_s1: MPSCNNConvolution // 32x32x128 input, kernels (1x1x128x256 = 32768 weights + 256 bias) s=1,p=0
    
    let conv4_1_dw: DepthwiseConvolutionKernel // 32x32x256 input, kernels (3x3x256 = 2304 weights + 256 bias) s=1,p=1
    let conv4_1_s1: MPSCNNConvolution // 32x32x256 input, kernels (1x1x256x256 = 65536 weights + 256 bias) s=1,p=0
    let conv4_2_dw: DepthwiseConvolutionKernel // 32x32x256 input, kernels (3x3x256 = 2304 weights + 256 bias) s=2,p=1
    let conv4_2_s1: MPSCNNConvolution // 32x32x256 input, kernels (1x1x256x512 = 131072 weights + 512 bias) s=1,p=0
    
    let conv5_1_dw: DepthwiseConvolutionKernel // 32x32x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=1,p=1
    let conv5_1_s1: MPSCNNConvolution // 32x32x512 input, kernels (1x1x512x512 = 262144 weights + 512 bias) s=1,p=0
    let conv5_2_dw: DepthwiseConvolutionKernel // 32x32x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=1,p=1
    let conv5_2_s1: MPSCNNConvolution // 32x32x512 input, kernels (1x1x512x512 = 262144 weights + 512 bias) s=1,p=0
    let conv5_3_dw: DepthwiseConvolutionKernel // 32x32x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=1,p=1
    let conv5_3_s1: MPSCNNConvolution // 32x32x512 input, kernels (1x1x512x512 = 262144 weights + 512 bias) s=1,p=0
    let conv5_4_dw: DepthwiseConvolutionKernel // 32x32x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=1,p=1
    let conv5_4_s1: MPSCNNConvolution // 32x32x512 input, kernels (1x1x512x512 = 262144 weights + 512 bias) s=1,p=0
    let conv5_5_dw: DepthwiseConvolutionKernel // 32x32x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=1,p=1
    let conv5_5_s1: MPSCNNConvolution // 32x32x512 input, kernels (1x1x512x512 = 262144 weights + 512 bias) s=1,p=0
    let conv5_6_dw: DepthwiseConvolutionKernel // 32x32x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=2,p=1
    let conv5_6_s1: MPSCNNConvolution // 32x32x512 input, kernels (1x1x512x1024 = 524288 weights + 1024 bias) s=1,p=0
    
    let conv6_1_dw: DepthwiseConvolutionKernel // 32x32x1024 input, kernels (3x3x1024 = 9216 weights + 1024 bias) s=1,p=1
    let conv6_1_s1: MPSCNNConvolution // 32x32x1024 input, kernels (1x1x1024x1024 = 1048576 weights + 1024 bias) s=1,p=0
    let fc7_nor: MPSCNNCrossChannelNormalization  //32x32x1024 input
    let fc7_nor_scale: ChannelScale               //32x32x1024 input
    let fc7_nor_score18: MPSCNNConvolution        //32x32x1024 input, kernels (1x1x1024x18)
    
    let pool6: MPSCNNPoolingAverage               // 32x32x1024 input ->1x1x1024 output
    let pool6_nor: MPSCNNCrossChannelNormalization//1x1x1024 input
    let pool6_nor_scale: ChannelScale             //1x1x1024 input
    let pool6_nor_score18: MPSCNNConvolution      //1x1x1024 input, kernels (1x1x1024x18)
    let pool6_nor_upscore18: Unpooling            //32x32x18, 1x1x18->32x32x18
    let score18: EltwiseKernel                    //32x32x18 input-> 32x32x18
    let score18_resize: MultiChannelResizeKernel  //32x32x18 input -> 256x256x18
    let result_maxout: Maxout   //for segmentation result 256x256x18 input -> 256x256x1
    let result_pallete:PalleteKernel //input 256x256x1 -> 256x256x3(RGB image)
    
    
    
    /* These MPSImage descriptors tell the network about the sizes of the data
     volumes that flow between the layers. */
    
    let input_id  = MPSImageDescriptor(channelFormat: .float16, width: 256, height: 256, featureChannels: 3)
    let conv1_id  = MPSImageDescriptor(channelFormat: .float16, width: 128, height: 128, featureChannels: 32)
    let conv2_1dw_id = MPSImageDescriptor(channelFormat: .float16, width: 128, height: 128, featureChannels: 32)
    let conv2_1s_id  = MPSImageDescriptor(channelFormat: .float16, width: 128, height: 128, featureChannels: 64)
    let conv2_2dw_id = MPSImageDescriptor(channelFormat: .float16, width: 64, height: 64, featureChannels: 64)
    let conv2_2s_id =  MPSImageDescriptor(channelFormat: .float16, width: 64, height: 64, featureChannels: 128)
    
    let conv3_1dw_id = MPSImageDescriptor(channelFormat: .float16, width: 64, height: 64, featureChannels: 128)
    let conv3_1s_id =  MPSImageDescriptor(channelFormat: .float16, width: 64, height: 64, featureChannels: 128)
    let conv3_2dw_id = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 128)
    let conv3_2s_id =  MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 256)
    
    let conv4_1dw_id = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 256)
    let conv4_1s_id  = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 256)
    let conv4_2dw_id = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 256)
    let conv4_2s_id  = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 512)
    
    let conv5_dw_id  = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 512)
    let conv5_s_id   = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 512)
    let conv5_6dw_id = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 512)
    let conv5_6s_id  = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 1024)
    
    let conv6_dw_id  = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 1024)
    let conv6_s_id   = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 1024)
    
    
    let fc7_nor_id = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 1024)
    let fc7_nor_scale_id = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 1024)
    
    let fc7_nor_score18_id = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 18)
    let pool6_id = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 1024)
    let pool6_nor_id = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 1024)
    let pool6_nor_scale_id = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 1024)
    
    let pool6_nor_score18_id = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 18)
    let pool6_nor_upscore18_id = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 18)
    
    let score18_id = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 18)
    let score18_resize_id = MPSImageDescriptor(channelFormat: .float16, width: 256, height: 256, featureChannels: 18)
    
    let maxout_img_id = MPSImageDescriptor(channelFormat: .float16, width: 256, height: 256, featureChannels: 1)
    
    let seg_img_id = MPSImageDescriptor(channelFormat: .float16, width: 256, height: 256, featureChannels: 4)
    
    let output_id = MPSImageDescriptor(channelFormat: .float16, width:   1, height:   1, featureChannels: 1000)
    
//    let t1:EltwiseKernel

    
    let labels = MobileNetsLabels()
    
    public init(device: MTLDevice) {
        print("Setting up neural network...")
        let startTime = CACurrentMediaTime()
        
        self.device = device
        commandQueue = device.makeCommandQueue()
        
        outputImage = MPSImage(device: device, imageDescriptor: output_id)
        
        seg_img = MPSImage(device: device, imageDescriptor: seg_img_id)
        
        // Before we pass an image into the network, we need to adjust its RGB
        // values. This is done with a custom compute kernel. Here we load that
        // kernel (from Shaders.metal) and set up the compute pipeline.
        do {
            let library = device.newDefaultLibrary()!
            let adjust_mean_rgb = library.makeFunction(name: "adjust_mean_rgb")
            pipelineRGB = try device.makeComputePipelineState(function: adjust_mean_rgb!)
            
            let adjust_mean_bgr = library.makeFunction(name: "adjust_mean_bgr")
            pipelineBGR = try device.makeComputePipelineState(function: adjust_mean_bgr!)
            
        } catch {
            fatalError("Error initializing compute pipeline")
        }
        
        // Uncomment this to test the network with all zero weights.
        //let blob = MobileNetsData()
        guard let path = Bundle.main.path(forResource: "parsenet_weights", ofType: "bat"),
            let blob = ParseNetData(path: path) else {
                fatalError("Error loading network parameters")
        }
        
        lanczos = MPSImageLanczosScale(device: device)
        lanczos_large = MPSImageLanczosScale(device: device)
        
        let relu = MPSCNNNeuronReLU(device: device, a: 0)
        
       // t1 = EltwiseKernel(device: device,featureChannels:32, neuronFilter: relu) //for test

        conv1_s2 = SlimMPSCNNConvolution(kernelWidth: 3,
                                         kernelHeight: 3,
                                         inputFeatureChannels: 3,
                                         outputFeatureChannels: 32,
                                         neuronFilter: relu,
                                         device: device,
                                         weights:blob.conv1_s2_w,
                                         bias: blob.conv1_s2_b,
                                         padding: true,
                                         strideXY: (2,2)
                                         )
        conv2_1_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 32,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv2_1_dw_w,
                                                biasTerms: blob.conv2_1_dw_b)
        
        conv2_1_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 32,
                                           outputFeatureChannels: 64,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv2_1_s1_w,
                                           bias: blob.conv2_1_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
                                           )
        conv2_2_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 64,
                                                strideInPixelsX: 2,
                                                strideInPixelsY: 2,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv2_2_dw_w,
                                                biasTerms: blob.conv2_2_dw_b)
        conv2_2_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 64,
                                           outputFeatureChannels: 128,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv2_2_s1_w,
                                           bias: blob.conv2_2_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        
        
        conv3_1_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 128,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv3_1_dw_w,
                                                biasTerms: blob.conv3_1_dw_b)
        conv3_1_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 128,
                                           outputFeatureChannels: 128,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv3_1_s1_w,
                                           bias: blob.conv3_1_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        conv3_2_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 128,
                                                strideInPixelsX: 2,
                                                strideInPixelsY: 2,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv3_2_dw_w,
                                                biasTerms: blob.conv3_2_dw_b)
        conv3_2_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                          kernelHeight: 1,
                                          inputFeatureChannels: 128,
                                          outputFeatureChannels: 256,
                                          neuronFilter: relu,
                                          device: device,
                                          weights:blob.conv3_2_s1_w,
                                          bias: blob.conv3_2_s1_b,
                                          padding: false,
                                          strideXY: (1,1)
        )
        
        conv4_1_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 256,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv4_1_dw_w,
                                                biasTerms: blob.conv4_1_dw_b)
        conv4_1_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 256,
                                           outputFeatureChannels: 256,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv4_1_s1_w,
                                           bias: blob.conv4_1_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        conv4_2_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 256,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv4_2_dw_w,
                                                biasTerms: blob.conv4_2_dw_b)
        conv4_2_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 256,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv4_2_s1_w,
                                           bias: blob.conv4_2_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )

        conv5_1_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_1_dw_w,
                                                biasTerms: blob.conv5_1_dw_b)
        conv5_1_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_1_s1_w,
                                           bias: blob.conv5_1_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        
        conv5_2_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_2_dw_w,
                                                biasTerms: blob.conv5_2_dw_b)
        conv5_2_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_2_s1_w,
                                           bias: blob.conv5_2_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        
        conv5_3_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_3_dw_w,
                                                biasTerms: blob.conv5_3_dw_b)
        conv5_3_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_3_s1_w,
                                           bias: blob.conv5_3_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        conv5_4_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_4_dw_w,
                                                biasTerms: blob.conv5_4_dw_b)
        conv5_4_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_4_s1_w,
                                           bias: blob.conv5_4_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        conv5_5_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_5_dw_w,
                                                biasTerms: blob.conv5_5_dw_b)
        conv5_5_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_5_s1_w,
                                           bias: blob.conv5_5_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        conv5_6_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_6_dw_w,
                                                biasTerms: blob.conv5_6_dw_b)
        conv5_6_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 1024,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_6_s1_w,
                                           bias: blob.conv5_6_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
   
        conv6_1_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 1024,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv6_1_dw_w,
                                                biasTerms: blob.conv6_1_dw_b)
        conv6_1_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 1024,
                                           outputFeatureChannels: 1024,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv6_1_s1_w,
                                           bias: blob.conv6_1_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        fc7_nor = makeNormalize(device: device)
        fc7_nor_scale = ChannelScale(device: device, featureChannels: 1024, scaleWeights: blob.fc7_nor_scale_w)
        fc7_nor_score18 = SlimMPSCNNConvolution(kernelWidth: 1,
                                                kernelHeight: 1,
                                                inputFeatureChannels: 1024,
                                                outputFeatureChannels: 18,
                                                neuronFilter: nil,
                                                device: device,
                                                weights:blob.fc7_nor_score18_w,
                                                bias: blob.fc7_nor_score18_b,
                                                padding: false,
                                                strideXY: (1,1)
        )
        
        pool6 = makePool(device: device)
        pool6_nor = makeNormalize(device: device)
        pool6_nor_scale = ChannelScale(device: device, featureChannels: 1024, scaleWeights: blob.pool6_nor_scale_w)
        pool6_nor_score18 = SlimMPSCNNConvolution(kernelWidth: 1,
                                                      kernelHeight: 1,
                                                      inputFeatureChannels: 1024,
                                                      outputFeatureChannels: 18,
                                                      neuronFilter: nil,
                                                      device: device,
                                                      weights:blob.pool6_nor_score18_w,
                                                      bias: blob.pool6_nor_score18_b,
                                                      padding: false,
                                                      strideXY: (1,1)
        )
        
        pool6_nor_upscore18 = Unpooling(device: device,
                                        featureChannels: 18,
                                        width: 32,
                                        height: 32)
        score18 = EltwiseKernel(device: device, featureChannels: 18)
        
        score18_resize = MultiChannelResizeKernel(device: device, featureChannels: 18)
        
        result_maxout = Maxout(device: device, featureChannels: 18)

        result_pallete = PalleteKernel(device: device, featureChannels: 1)
        
        softmax = MPSCNNSoftMax(device: device)
        
        let endTime = CACurrentMediaTime()
        print("Elapsed time: \(endTime - startTime) sec")
    }
    
    /* Performs the inference step. This takes the input image, converts it into
     the format the network expects, then feeds it into the network. The result
     is a 1000-element vector of probabilities. Returns the 5 ImageNet classes
     with the highest predicted probability values. */
    public func predict(image inputImage: MPSImage, bgr: Bool) -> [Prediction] {
        let startTime = CACurrentMediaTime()
        
        autoreleasepool{
            let commandBuffer = commandQueue.makeCommandBuffer()
            
            // This lets us squeeze some extra speed out of Metal.
            MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: [
                input_id, conv1_id, conv2_1dw_id, conv2_1s_id, conv2_2dw_id, conv2_2s_id, conv3_1dw_id,conv3_1s_id,
                conv3_2dw_id,conv3_2s_id,conv4_1dw_id,conv4_1s_id,conv4_2dw_id,conv4_2s_id,conv5_dw_id,conv5_s_id,
                conv5_6dw_id,conv5_6s_id,conv6_dw_id,conv6_s_id,fc7_nor_id,fc7_nor_score18_id,pool6_id,pool6_nor_id,pool6_nor_score18_id,score18_id,score18_resize_id, output_id ])
            
            // Scale the input image to 256x256 pixels.
            let img1 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: input_id)
            lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: img1.texture)
           
            //let img2 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: input_id)
            let img2 = MPSImage(device: device, imageDescriptor: input_id)
            
            // Adjust the RGB values of each pixel to be in the range -128...127
            // by subtracting the "mean pixel". If the input texture is RGB, this
            // also swaps the R and B values because the model expects BGR pixels.
            // As far as I can tell there is no MPS shader that can do these things,
            // so we use a custom compute kernel.
            let encoder = commandBuffer.makeComputeCommandEncoder()
            encoder.setComputePipelineState(bgr ? pipelineBGR : pipelineRGB)
            encoder.setTexture(img1.texture, at: 0)
            encoder.setTexture(img2.texture, at: 1)
            let threadsPerGroups = MTLSizeMake(8, 8, 1)
            let threadGroups = MTLSizeMake(img2.texture.width / threadsPerGroups.width,
                                           img2.texture.height / threadsPerGroups.height, 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
            encoder.endEncoding()
            img1.readCount -= 1    // see MPSTemporaryImage docs why this is needed
        
//            //for eltwise test
//            let testImage = MPSImage(device: device, imageDescriptor: input_id)  //for test
//            t1.encode(commandBuffer: commandBuffer, sourceImage1: img2, sourceImage2: img2, destinationImage: testImage)
            
            // Now we take the output from our custom shader and pass it through the
            // layers of the neural network. For each layer we use a new "temporary"
            // MPSImage to hold the results.
            
            
//            let conv1_s2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv1_id)
//            conv1_s2_img.readCount = 2
            let conv1_s2_img = MPSImage(device: device, imageDescriptor: conv1_id)
            conv1_s2.encode(commandBuffer: commandBuffer, sourceImage: img2, destinationImage: conv1_s2_img)
            
            let conv2_1dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_1dw_id)
//            let conv2_1dw_img = MPSImage(device: device, imageDescriptor: conv2_1dw_id)
            conv2_1_dw.encode(commandBuffer: commandBuffer, sourceImage: conv1_s2_img, destinationImage: conv2_1dw_img)
            let conv2_1s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_1s_id)
            conv2_1_s1.encode(commandBuffer: commandBuffer, sourceImage: conv2_1dw_img, destinationImage: conv2_1s_img)
            
            //for eltwise test
            //let testImage = MPSImage(device: device, imageDescriptor: conv2_1dw_id)  //for test
            //t1.encode(commandBuffer: commandBuffer, sourceImage1: conv1_s2_img, sourceImage2: conv1_s2_img, destinationImage: testImage)
            
            let conv2_2dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_2dw_id)
            conv2_2_dw.encode(commandBuffer: commandBuffer, sourceImage: conv2_1s_img, destinationImage: conv2_2dw_img)
            let conv2_2s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_2s_id)
            conv2_2_s1.encode(commandBuffer: commandBuffer, sourceImage: conv2_2dw_img, destinationImage: conv2_2s_img)
            
            let conv3_1dw_img  = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_1dw_id)
            conv3_1_dw.encode(commandBuffer: commandBuffer, sourceImage: conv2_2s_img, destinationImage: conv3_1dw_img)
            let conv3_1s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_1s_id)
            conv3_1_s1.encode(commandBuffer: commandBuffer, sourceImage: conv3_1dw_img, destinationImage: conv3_1s_img)
            let conv3_2dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_2dw_id)
            conv3_2_dw.encode(commandBuffer: commandBuffer, sourceImage: conv3_1s_img, destinationImage: conv3_2dw_img)
            let conv3_2s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_2s_id)
            conv3_2_s1.encode(commandBuffer: commandBuffer, sourceImage: conv3_2dw_img, destinationImage: conv3_2s_img)
            
            let conv4_1dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_1dw_id)
            conv4_1_dw.encode(commandBuffer: commandBuffer, sourceImage: conv3_2s_img, destinationImage: conv4_1dw_img)
            let conv4_1s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_1s_id)
            conv4_1_s1.encode(commandBuffer: commandBuffer, sourceImage: conv4_1dw_img, destinationImage: conv4_1s_img)
            let conv4_2dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_2dw_id)
            conv4_2_dw.encode(commandBuffer: commandBuffer, sourceImage: conv4_1s_img, destinationImage: conv4_2dw_img)
            let conv4_2s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_2s_id)
            conv4_2_s1.encode(commandBuffer: commandBuffer, sourceImage: conv4_2dw_img, destinationImage: conv4_2s_img)
            
            let conv5_1dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_dw_id)
            conv5_1_dw.encode(commandBuffer: commandBuffer, sourceImage: conv4_2s_img, destinationImage: conv5_1dw_img)
            let conv5_1s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_s_id)
            conv5_1_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_1dw_img, destinationImage: conv5_1s_img)
            let conv5_2dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_dw_id)
            conv5_2_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_1s_img, destinationImage: conv5_2dw_img)
            let conv5_2s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_s_id)
            conv5_2_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_2dw_img, destinationImage: conv5_2s_img)
            let conv5_3dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_dw_id)
            conv5_3_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_2s_img, destinationImage: conv5_3dw_img)
            let conv5_3s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_s_id)
            conv5_3_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_3dw_img, destinationImage: conv5_3s_img)
            let conv5_4dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_dw_id)
            conv5_4_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_3s_img, destinationImage: conv5_4dw_img)
            let conv5_4s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_s_id)
            conv5_4_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_4dw_img, destinationImage: conv5_4s_img)
            let conv5_5dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_dw_id)
            conv5_5_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_4s_img, destinationImage: conv5_5dw_img)
            let conv5_5s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_s_id)
            conv5_5_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_5dw_img, destinationImage: conv5_5s_img)
            let conv5_6dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_6dw_id)
            conv5_6_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_5s_img, destinationImage: conv5_6dw_img)
            let conv5_6s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_6s_id)
            conv5_6_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_6dw_img, destinationImage: conv5_6s_img)
            
            let conv6_dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv6_dw_id)
            conv6_1_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_6s_img, destinationImage: conv6_dw_img)
            //let conv6_s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv6_s_id)
            //conv6_s_img.readCount = 2
            let conv6_s_img = MPSImage(device: device, imageDescriptor: conv6_s_id)
            conv6_1_s1.encode(commandBuffer: commandBuffer, sourceImage: conv6_dw_img, destinationImage: conv6_s_img)
            
            //let fc7_nor_img =  MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc7_nor_id)
            let fc7_nor_img =  MPSImage(device: device, imageDescriptor: fc7_nor_id)
            fc7_nor.encode(commandBuffer: commandBuffer, sourceImage: conv6_s_img, destinationImage: fc7_nor_img)
            //let fc7_nor_scale_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc7_nor_scale_id)
            let fc7_nor_scale_img = MPSImage(device: device, imageDescriptor: fc7_nor_scale_id)
            fc7_nor_scale.encode(commandBuffer: commandBuffer, sourceImage: fc7_nor_img, destinationImage: fc7_nor_scale_img)
//            let fc7_nor_score18_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc7_nor_score18_id)
//            fc7_nor_score18_img.readCount = 2
            let fc7_nor_score18_img = MPSImage(device: device, imageDescriptor: fc7_nor_score18_id)
            fc7_nor_score18.encode(commandBuffer: commandBuffer, sourceImage: fc7_nor_scale_img, destinationImage: fc7_nor_score18_img)
            
            
            //let pool6_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool6_id)
            let pool6_img = MPSImage(device: device, imageDescriptor: pool6_id)
            pool6.encode(commandBuffer: commandBuffer, sourceImage: conv6_s_img, destinationImage: pool6_img)
            let pool6_nor_img = MPSImage(device: device, imageDescriptor: pool6_nor_id)
            //let pool6_nor_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool6_nor_id)
            pool6_nor.encode(commandBuffer: commandBuffer, sourceImage: pool6_img, destinationImage: pool6_nor_img)
            //let pool6_nor_scale_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool6_nor_scale_id)
            let pool6_nor_scale_img = MPSImage(device: device, imageDescriptor: pool6_nor_scale_id)
            pool6_nor_scale.encode(commandBuffer: commandBuffer, sourceImage: pool6_nor_img, destinationImage: pool6_nor_scale_img)
            //let pool6_nor_score18_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool6_nor_score18_id)
            let pool6_nor_score18_img = MPSImage(device: device, imageDescriptor: pool6_nor_score18_id)
            pool6_nor_score18.encode(commandBuffer: commandBuffer, sourceImage: pool6_nor_scale_img, destinationImage: pool6_nor_score18_img)
            
            //let pool6_nor_upscore18_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool6_nor_upscore18_id)
            let pool6_nor_upscore18_img = MPSImage(device: device, imageDescriptor: pool6_nor_upscore18_id)
            pool6_nor_upscore18.encode(commandBuffer: commandBuffer, sourceImage: pool6_nor_score18_img, destinationImage: pool6_nor_upscore18_img)
            
            let score18_img = MPSImage(device: device, imageDescriptor: score18_id)
            score18.encode(commandBuffer: commandBuffer, sourceImage1: fc7_nor_score18_img, sourceImage2: pool6_nor_upscore18_img, destinationImage: score18_img)
            
            let score18_img_resize = MPSImage(device: device, imageDescriptor: score18_resize_id)
            score18_resize.encode(commandBuffer: commandBuffer, sourceImage: score18_img, destinationImage: score18_img_resize)
            
            let maxout_img = MPSImage(device: device, imageDescriptor: maxout_img_id)
            result_maxout.encode(commandBuffer: commandBuffer, sourceImage: score18_img_resize, destinationImage: maxout_img)
            
            result_pallete.encode(commandBuffer: commandBuffer, sourceImage: maxout_img, destinationImage: seg_img)
            
//            let score18_large_img = MPSImage(device: device, imageDescriptor: score18_large_id)
//            print(score18_large_img.featureChannels)
            
//            lanczos_large.encode(commandBuffer: commandBuffer, sourceTexture: score18_img.texture, destinationTexture: score18_large_img.texture)
            //lanczos_large.encode(commandBuffer: commandBuffer, sourceTexture: img1.texture, destinationTexture: img2.texture)
            
//            lanczos_large.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: img1.texture)
            
            
            // Tell the GPU to start and wait until it's done.
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            print(conv1_s2_img.toFloatArray()[0...4096])
            //print(testImage.toFloatArray()[100176...100186])
//            print(pool6_nor_upscore18_img.toFloatArray())
//            print(pool6_nor_score18_img.toFloatArray())
//            print(pool6_img.toFloatArray()[0...100])
//            print(pool6_nor_img.toFloatArray()[0...100])
//            print(pool6_nor_scale_img.toFloatArray()[0...100])
//            print(conv6_s_img.toFloatArray()[0...4095])
//            print(fc7_nor_img.toFloatArray()[0...4095])
//            print(fc7_nor_scale_img.toFloatArray()[0...4095])
            //print(fc7_nor_score18_img.toFloatArray()[0...4095])
//            print(pool6_nor_upscore18_img.toFloatArray()[0...4096])
//            print(score18_img.toFloatArray()[0...110])
//            print(maxout_img.toFloatArray())
 //           print(seg_img.toFloatArray())
        
        }
        
        
        // Convert the texture from outputImage into something we can use from
        // Swift and then find the ImageNet classes with the highest probability.
        let result = self.labels.top5Labels(prediction: self.outputImage.toFloatArray())
        
        let endTime = CACurrentMediaTime()
        print("Elapsed time: \(endTime - startTime) sec")
        
        return result
    }
}

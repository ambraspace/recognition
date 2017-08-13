package com.ambraspace.recognition.dl4j;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Net51detection {

    private static final Logger log = LoggerFactory.getLogger(Net51detection.class);
	
	public static void main(String[] args) {
		
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		uiServer.attach(statsStorage);
		
		double bestRecall = 0.5f;
		
		int imageWidth = 51;
		int imageHeight = 51;
		int channels = 1;
		int outputNum = 2;
		int numEpochs = 1000;
		int batchSize = 225;
		int rngSeed = 123;
		Random randNumGen = new Random(rngSeed);
		int iterations = 1;

		
        File parentDir = new File("/home/ambra/Deep Learning/LFW/training/detection-51");
        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, new String[]{"png"}, randNumGen);
        
        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

//        BalancedPathFilter pathFilter = new BalancedPathFilter(
//        		randNumGen,
//        		new String[]{"png"},
//        		labelMaker, 3000);
        RandomPathFilter pathFilter = new RandomPathFilter(randNumGen, new String[]{"png"});
        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 90, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];
        
        ImageRecordReader rr1 = new ImageRecordReader(imageHeight, imageHeight, channels, labelMaker);
        //ImageRecordReader rr1s = new ImageRecordReader(13, 13, channels, labelMaker);
        ImageRecordReader rr2 = new ImageRecordReader(imageHeight, imageWidth, channels, labelMaker);
        //ImageRecordReader rr2s = new ImageRecordReader(13, 13, channels, labelMaker);

        try {
			rr1.initialize(trainData);
			//rr1s.initialize(trainData);
	        rr2.initialize(testData);
	        //rr2s.initialize(testData);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

        
//        MultiDataSetIterator trainDataIterator = new RecordReaderMultiDataSetIterator.Builder(1)
//        		.addReader("rr1", rr1)
//        		.addReader("rr1s", rr1s)
//        		.addInput("rr1", 0, 0)
//        		.addInput("rr1s", 0, 0)
//        		.addOutputOneHot("rr1s", 1, outputNum)
//        		.build();
//        
//        MultiDataSet mds = trainDataIterator.next();
//        mds.toString();
//        
//        MultiDataSetIterator testDataIterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
//        		.addReader("rr2", rr2)
//        		.addReader("rr2s", rr2s)
//        		.addInput("rr2", 0, 0)
//        		.addInput("rr2s", 0, 0)
//        		.addOutput("rr2", 1, 1)
//        		.build();
//
//        MultiDataSetPreProcessor ipps = new MultiDataSetPreProcessor() {
//
//        	ImagePreProcessingScaler ip = new ImagePreProcessingScaler();
//        	
//			@Override
//			public void preProcess(MultiDataSet multiDataSet) {
//				for (int i=0; i<multiDataSet.numFeatureArrays(); i++) {
//					//ip.preProcess(multiDataSet.getFeatures(i));
//				}
//			}
//		};
        
        
        
        ImagePreProcessingScaler ipps = new ImagePreProcessingScaler();
        
        DataSetIterator trainDataIterator = new RecordReaderDataSetIterator(rr1, batchSize);
        DataSetIterator testDataIterator = new RecordReaderDataSetIterator(rr2, batchSize);
        
        trainDataIterator.setPreProcessor(ipps);
        testDataIterator.setPreProcessor(ipps);

        MultiLayerNetwork model = null;
        
        File modelFile = new File("model.zip");
        if (modelFile.exists()) {
        	log.info("Loading prepared model.");
        	try {
				model = ModelSerializer.restoreMultiLayerNetwork(modelFile, true);
			} catch (IOException e) {
				log.error("Unable to load the model!");
			}
        }
		

        log.info("Build model....");
//        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
//                .learningRate(0.05)//.biasLearningRate(0.02)
//                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
//            	.graphBuilder()
//            	.addInputs("input1", "input2")
//            	.addLayer("Conv1", new ConvolutionLayer.Builder(5, 5)
//                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
//                        .nIn(channels)
//                        .stride(1, 1)
//                        .nOut(64)
//                        .activation("relu")
//                        .build(), "input1")
//            	.addLayer("MaxPool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(3,3)
//                        .stride(2,2)
//                        .build(), "Conv1")
//            	.addLayer("FCL1", new DenseLayer.Builder().activation("relu")
//                        .nOut(128).build(), "MaxPool1")
//            	.addLayer("Conv2", new ConvolutionLayer.Builder(3, 3)
//                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
//                        .nIn(channels)
//                        .stride(1, 1)
//                        .nOut(16)
//                        .activation("relu")
//                        .build(), "input2")
//            	.addLayer("MaxPool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(3,3)
//                        .stride(2,2)
//                        .build(), "Conv2")
//            	.addLayer("FCL2", new DenseLayer.Builder().activation("relu")
//                        .nOut(16).build(), "MaxPool2")
//            	.addVertex("merge", new MergeVertex(), "FCL1", "FCL2")
//            	.addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(outputNum)
//                        .activation("softmax")
//                        .build(), "merge")
//            	.setOutputs("output")
//            	.setInputTypes(
//            			InputType.convolutional(imageHeight, imageWidth, channels),
//            			InputType.convolutional(13, 13, channels))
//            	.build();
//        
//        ComputationGraph network = new ComputationGraph(conf);
//        network.init();

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .iterations(iterations)
                .regularization(true).l2(0.005)
                .learningRate(0.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        //.activation("identity")
                        .build())
                .layer(2, new LocalResponseNormalization.Builder()
                		//.alpha(0)
                		//.beta(0)
                		//.k(0)
                		.n(9)//.activation("identity")
                		.build())
                .layer(3, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(4, new LocalResponseNormalization.Builder()
                		//.alpha(0)
                		//.beta(0)
                		//.k(0)
                		.n(9)//.activation("identity")
                		.build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        //.activation("identity")
                        .build())
                .layer(6, new DenseLayer.Builder()
                		.activation(Activation.RELU)
                        .nOut(256).build())
                .layer(7, new OutputLayer.Builder(
                		new LossBinaryXENT(new NDArray(new double[][]{{0.1, 1.0}})))
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false);

        builder.setInputType(InputType.convolutional(imageHeight, imageWidth, channels));
        
        MultiLayerConfiguration conf = builder.build();
       
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        
        if (model!=null) { 
        	network.setParams(model.params());
        }
        
        log.info("Train model...");
//        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
        network.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<numEpochs; i++ ) {
        	Long start = System.currentTimeMillis();
        	network.fit(trainDataIterator);
            Long stop = System.currentTimeMillis();
            log.info("*** Completed epoch {} ***", i);
            log.info("It took " + (stop-start)/1000 + " seconds.");
            log.info("Evaluate model...");
            Evaluation eval = new Evaluation(outputNum);
            while(testDataIterator.hasNext()){
                DataSet ds = testDataIterator.next();
                INDArray output = network.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }
            log.info(eval.stats());
            testDataIterator.reset();
            
//            if (eval.recall()>bestRecall) {
	          	bestRecall=eval.recall();
	            try {
	            	ModelSerializer.writeModel(network, String.format("%05d-%4.4f", i, bestRecall)+".zip", true);
				} catch (IOException e) {
					log.error("Can not save model!");
				}
//            }

        }
        log.info("****************Example finished********************");

	}

	
}

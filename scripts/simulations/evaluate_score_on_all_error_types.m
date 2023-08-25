close all;
clear all;

path_to_root = ...;
path_to_utrack3D_codebase = ...;
path_to_temporary_folder = ...;

%% Generate tree of paths from the repo's root
addpath(genpath(path_to_utrack3D_codebase));
% Check that the code has been loaded correctly
if(isempty(which('MovieData')))
	error('The code folder must be loaded first.');
end


% Start parallel pool for parallel computing
try
	parpool(2);
catch 
	disp('Parallel pool running');
end

saveFolder = '/tmp/utrack_jules/test_myfiles'; % Output folder for movie metadata and results
mkClrDir(saveFolder);


pixelSize_ = 1; % Anisotropy in X/Y
pixelSizeZ_ = 1; % Anisotropy in Z
timeInterval_ = 1;


SR = 10;
all_names = ["fn" "fp" "merge" "split"];


for name = all_names
    
    mkdir(strcat(path_to_root,'/simulations/utrack/',name,'/'));
    
    for ind_test = 0:1:9
    
        disp(["current name" name])
        disp(["current test" ind_test])
        
        mkdir(strcat(path_to_root,'/simulations/utrack/',name,'/',num2str(ind_test)));
        

        for rate = 0:5:45


            if rate == 5 || rate == 0 
                folder_name = strcat(name,'/' ,num2str(ind_test),'/', 'coords_',name,'_00',num2str(rate),'/');
            else
                folder_name = strcat(name,'/' ,num2str(ind_test),'/', 'coords_',name,'_0',num2str(rate),'/');
            end

            path_to_inputs = strcat(path_to_root,'/simulations/coords/',folder_name);
            path_toresults = strcat(path_to_root,'/simulations/utrack/',folder_name);
            
            mkdir(path_to_results);


        %% Detection parameterization and processing
            directory_infos = dir(strcat(path_to_inputs, '*.csv'));
            nFrames = length(directory_infos);

            dCell=cell(1,nFrames);  
            ZXRatio=pixelSizeZ_/pixelSize_;

            for fIdx=1:nFrames
                d=Detections();

                CSV = csvread(strcat(path_to_inputs, directory_infos(fIdx).name));

                % go from ZYX format to XYZ
                CSV = CSV(:,[3,2,1]);

                % corrupt coords with controlled noise
                stdCorruption = 0.1;
                stdPixel = 0.5;
                CSV = CSV + stdCorruption .* randn(size(CSV));

                d=d.initFromPosMatrices(CSV,sqrt(stdCorruption^2 + stdPixel^2) .* ones(size(CSV)));
                %d=d.initFromPosMatrices(CSV,stdCorruption .* ones(size(CSV)));
                dCell{fIdx}=d;
            end
            movieInfo=[dCell{:}];


            %% cost matrix for gap closing
            gapCloseParam.timeWindow = 3; %maximum allowed time gap (in frames) %between a track segment end and a track segment start that allows linking them.
            gapCloseParam.mergeSplit = 0; %1 if merging and splitting are to be considered, 2 if only merging is to be considered, 3 if only splitting is to be considered, 0 if no merging or splitting are to be considered.
            gapCloseParam.minTrackLen = 2; %minimum length of track segments from linking to be used in gap closing.



            %optional input:
            gapCloseParam.diagnostics = 0; %1 to plot a histogram of gap lengths in the end; 0 or empty otherwise.

            %% cost matrix for frame-to-frame linking
            %function name
            costMatrices(1).funcName = 'costMatRandomDirectedSwitchingMotionLink';

            %parameters
            parameters.linearMotion = 1; %use linear motion Kalman filter.

            parameters.minSearchRadius = SR;%4;%2;% %minimum allowed search radius. The search radius is calculated on the spot in the code given a feature's motion parameters. If it happens to be smaller than this minimum, it will be increased to the minimum.
            parameters.maxSearchRadius = SR+0.1;%10;%6;% %maximum allowed search radius. Again, if a feature's calculated search radius is larger than this maximum, it will be reduced to this maximum.
            parameters.brownStdMult = 3; %multiplication factor to calculate search radius from standard deviation.

            parameters.useLocalDensity = 0; %1 if you want to expand the search radius of isolated features in the linking (initial tracking) step.
            parameters.nnWindow = gapCloseParam.timeWindow; %number of frames before the current one where you want to look to see a feature's nearest neighbor in order to decide how isolated it is (in the initial linking step).

            parameters.kalmanInitParam = []; %Kalman filter initialization parameters.
            parameters.kalmanInitParam.searchRadiusFirstIteration = SR; %Kalman filter initialization parameters.

            %optional input
            parameters.diagnostics = 0; %if you want to plot the histogram of linking distances up to certain frames, indicate their numbers; 0 or empty otherwise. Does not work for the first or last frame of a movie.

            costMatrices(1).parameters = parameters;




            %function name
            costMatrices(2).funcName = 'costMatRandomDirectedSwitchingMotionCloseGaps';

            %parameters needed all the time
            parameters.linearMotion = 0; %use linear motion Kalman filter.

            parameters.minSearchRadius = 0; %minimum allowed search radius.
            parameters.maxSearchRadius = 0.0001; %maximum allowed search radius.
            parameters.brownStdMult = 3*ones(gapCloseParam.timeWindow,1); %multiplication factor to calculate Brownian search radius from standard deviation.

            %power for scaling the Brownian search radius with time, before and
            %after timeReachConfB (next parameter). Note that it is only the gap
            %value which is powered, then we have brownStdMult*powered_gap*sig*sqrt(dim)
            parameters.brownScaling = [0.25 0.01];
            % parameters.timeReachConfB = 3; %before timeReachConfB, the search radius grows with time with the power in brownScaling(1); after timeReachConfB it grows with the power in brownScaling(2).
            parameters.timeReachConfB = gapCloseParam.timeWindow; %before timeReachConfB, the search radius grows with time with the power in brownScaling(1); after timeReachConfB it grows with the power in brownScaling(2).

            parameters.ampRatioLimit = [0.7 4]; %for merging and splitting. Minimum and maximum ratios between the intensity of a feature after merging/before splitting and the sum of the intensities of the 2 features that merge/split.

            parameters.lenForClassify = 5; %minimum track segment length to classify it as linear or random.

            parameters.useLocalDensity = 0; %1 if you want to expand the search radius of isolated features in the gap closing and merging/splitting step.
            parameters.nnWindow = gapCloseParam.timeWindow; %number of frames before/after the current one where you want to look for a track's nearest neighbor at its end/start (in the gap closing step).

            parameters.linStdMult = 1*ones(gapCloseParam.timeWindow,1); %multiplication factor to calculate linear search radius from standard deviation.

            parameters.linScaling = [0.25 0.01]; %power for scaling the linear search radius with time (similar to brownScaling).
            % parameters.timeReachConfL = 4; %similar to timeReachConfB, but for the linear part of the motion.
            parameters.timeReachConfL = gapCloseParam.timeWindow; %similar to timeReachConfB, but for the linear part of the motion.

            parameters.maxAngleVV = 30; %maximum angle between the directions of motion of two tracks that allows linking them (and thus closing a gap). Think of it as the equivalent of a searchRadius but for angles.

            %optional; if not input, 1 will be used (i.e. no penalty)
            parameters.gapPenalty = 1.5; %penalty for increasing temporary disappearance time (disappearing for n frames gets a penalty of gapPenalty^n).

            %optional; to calculate MS search radius
            %if not input, MS search radius will be the same as gap closing search radius
            parameters.resLimit = []; %resolution limit, which is generally equal to 3 * point spread function sigma.

            costMatrices(2).parameters = parameters;


            %% Kalman filter function names

            kalmanFunctions.reserveMem  = 'kalmanResMemLM';
            kalmanFunctions.initialize  = 'kalmanInitLinearMotion';
            kalmanFunctions.calcGain    = 'kalmanGainLinearMotion';
            kalmanFunctions.timeReverse = 'kalmanReverseLinearMotion';

            schemeName='U_track';
            saveResults.dir =  [saveFolder filesep schemeName];
            %mkdir(saveResults.dir);


            probDim=3;

            verbose=0;

            %% Run tracker
            [tracksFinal,kalmanInfoLink,errFlag,trackabilityData] = ...
                trackCloseGapsKalmanSparse(movieInfo.getStruct(),costMatrices, ... 
                                           gapCloseParam,kalmanFunctions ,...
                                           probDim,saveResults,verbose,'estimateTrackability',true);

            tracks=TracksHandle(tracksFinal);


            %% Extract tracks and trackability data and dump them to .json 
            %tracks = TracksHandle(processTrack.loadChannelOutput(1)); 

            encoded = jsonencode(tracks);

            fid = fopen(strcat(path_to_results,'/','tracks.json'),'w');
            fprintf(fid,'%s',encoded);
            fclose(fid);


            segTrackability=cell(1,length(tracksFinal));
            for tIdx=1:length(tracks)
                tr=(tracks(tIdx));
                nonGap=~tr.gapMask();
                nonGap=nonGap(1:end-1);  % T-1 segments.
                linkIdx=find(nonGap);
                segTrackability{tIdx}=nan(size(nonGap));
                segTrackability{tIdx}(linkIdx)=arrayfun(@(pIdx) trackabilityData.trackabilityCost{tr.f(pIdx)+1}(tr.tracksFeatIndxCG(pIdx)),linkIdx);
            end

            encoded = jsonencode(segTrackability);

            fid = fopen(strcat(path_to_results,'/','trackability.json'),'w');
            fprintf(fid,'%s',encoded);
            fclose(fid);

        end % N_tests

    end % rate

end % name



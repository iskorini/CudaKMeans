int kMeansSequential(float *devData) {
	float *hostData, *mean, *oldMean;
	short *assignment;
	hostData = (float*) malloc(N * COMPONENTS * sizeof(float));
	mean = (float*) malloc(K * COMPONENTS * sizeof(float));
	assignment = (short*) malloc(N * sizeof(short));
	oldMean = (float*) malloc(K * COMPONENTS * sizeof(float));
	for (int i = 0; i < K; i++) {
		assignment[i] = i;
		for (int j = 0; j < COMPONENTS; j++) {
			mean[i * COMPONENTS + j] = hostData[i * COMPONENTS + j];
		}
	}
	int iter = 0;
	bool stopCriterion = false;
	while (!stopCriterion) {
		for (int i = 0; i < N; i++) {
			float minDistance = 999999.9;
			short minIndex = -1;
			float distance = 0;
			for (int z = 0; z < K; z++) {
				distance = 0;
				for (int j = 0; j < COMPONENTS; j++) {
					distance +=pow(
							(hostData[i * COMPONENTS + j]
									- mean[z * COMPONENTS + j]),2);
				}
				if (distance < minDistance) {
					minIndex = z;
					minDistance = distance;
				}
			}
			assignment[i] = minIndex;
		}
		for (int i = 0; i < K; i++) {
			int numberOfData = 0;
			float *arraySum = (float*) malloc(COMPONENTS * sizeof(float));
			for (int x = 0; x<COMPONENTS; x++) {
				arraySum[x] = 0.0; 
			}
			for (int j = 0; j < N; j++) {
				if (assignment[j] == i) {
					numberOfData++;
					for (int x = 0; x < COMPONENTS; x++) {
						arraySum[x] += hostData[j * COMPONENTS + x];
					}
				}
			}
			for (int j = 0; j < COMPONENTS; j++) {
				oldMean[i * COMPONENTS + j] = mean[i * COMPONENTS + j];
				mean[i * COMPONENTS + j] = arraySum[j] / numberOfData;
			}
		}
		stopCriterion = true;
		for (int i = 0; i < K * COMPONENTS; i++) {
			if (abs(mean[i] - oldMean[i]) > TOL) {
				stopCriterion = false;
				break;
			}
		}
	}
	return 0;
}
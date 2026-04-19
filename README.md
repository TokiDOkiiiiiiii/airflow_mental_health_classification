# Aiflow for MentalHealth Classification

## Workflow function is defined in dags/src/workflow

### List

Step1. Clean data
Step2. Check data drift
Step3. Retrain model
Step4. Evaluate and promote model

## Utility function

1. dataset -> class for dataloader
2. preprocessing -> function for data cleaning
3. testing -> function for model testing
4. wrapper -> pyfunc deployment flavor model class

## Deployment

1. delete -> delete endpoint function and script
2. deploy -> deploy new endpoint function and scirpt
3. inference -> inference script
4. update -> update the endpoint with new model

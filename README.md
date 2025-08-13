# AWS-SimuLearn-Machine-Learning-portfolio
Portfolio of 25 AWS SimuLearn machine learning projects plus self-initiated DIY solutions. Covers SageMaker, Rekognition, Comprehend, Textract, Transcribe, and other AWS AI services. Includes custom ML model training, deployment, reinforcement learning, computer vision, NLP, and generative AI integrations.

**AWS SimuLearn: Set Up an ML Environment** **|** [GitHub Repo](https://tinyurl.com/john0777)

**Summary:**  
Deployed and configured a full Amazon SageMaker ML environment (Domain + Studio), ran sample notebooks, and integrated serverless tooling to perform real-time inference.

**Key learning outcomes (brief):**
- Deployed an **Amazon SageMaker Domain** and launched **SageMaker Studio** for notebook-based ML development.  
- Cloned and ran sample ML projects from GitHub to explore end-to-end training and inference workflows.  
- Created and managed a **SageMaker inference endpoint** and changed instance type to **ml.m5.xlarge** for production-like inference.  
- Configured and tested an **AWS Lambda** function to invoke the SageMaker endpoint (real-time inference).  
- Performed endpoint lifecycle tasks (create/update/invoke) and validated inference responses.  
- **DIY:** Manually created a new `ml.m5.xlarge` endpoint and reran the Lambda invocation to validate the solution end-to-end.

**Tools / Services:** Amazon SageMaker (Studio, endpoints), AWS Lambda, GitHub, (S3 for data/artifacts).

**AWS SimuLearn: Fine-Tuning an LLM on Amazon SageMaker** **|** [GitHub Repo](https://tinyurl.com/john07777)

**Summary:**  
Hands-on fine-tuning of a pretrained large language model using SageMaker notebooks, deploying the tuned model to a SageMaker endpoint, and integrating it into an application for real-time inference.

**Key learning outcomes (brief):**
- Imported and ran Jupyter notebooks in **SageMaker Studio** to prepare code and training pipelines.  
- Prepared training datasets and applied fine-tuning techniques for a pretrained LLM (data prep, tokenization, training loop, checkpointing, and evaluation).  
- Deployed the fine-tuned model to a **SageMaker inference endpoint** and validated model behavior.  
- Integrated the endpoint into a practical application and updated serverless code to call the new endpoint (Lambda-based invocation).  
- Performed endpoint lifecycle tasks (create, update, test) and validated deployment against expected responses.

**DIY:**  
Fine-tuned the model using a custom dataset derived from **Amazon Bedrock FAQ** data, deployed the resulting model to a SageMaker endpoint, updated the AWS Lambda client to use the new endpoint, and tested the sample application end-to-end.

**Tools / Services:** Amazon SageMaker (Studio, endpoints), Jupyter notebooks, AWS Lambda, Amazon Bedrock (data source).

**AWS SimuLearn: Chatbots with a Large Language Model (LLM)** **|** [GitHub Repo](https://tinyurl.com/johnvanguri)

**Summary:**  
Built and deployed an LLM-backed chatbot using SageMaker JumpStart and real-time inference. Integrated the model with Amazon Lex and AWS Lambda (via LangChain) to create a production-style conversational flow with fallback handling and end-to-end validation.

**Key learning outcomes (brief):**
- Used **SageMaker JumpStart** to discover and deploy pretrained LLMs (e.g., FLAN-T5-XL) for NLP tasks.  
- Deployed models to **SageMaker real-time inference endpoints** and validated endpoint behavior.  
- Implemented chatbot integration: **Amazon Lex V2** for conversational interface, **AWS Lambda** for backend logic, and **LangChain** to connect Lex/Lambda to the SageMaker endpoint.  
- Configured a **Lex fallback intent** to gracefully handle unmatched user inputs.  
- Performed production-like endpoint lifecycle tasks: delete/redeploy endpoints, update environment variables, and test invocations.  
- **DIY:** Deployed a custom Hugging Face model (`huggingface-text2text-t5-one-line-summary`), uploaded `diy_results.txt` to an S3 bucket, and updated the Lambda environment variable to point to the new endpoint.

**Tools / Services:** Amazon SageMaker (JumpStart, endpoints), Amazon Lex V2, AWS Lambda, LangChain, Amazon S3, Hugging Face model IDs.

**AWS SimuLearn: Bring Your Own Model (BYOM)** **|** [GitHub Repo](https://tinyurl.com/johnn0077)

**Summary:**  
Learned how to containerize and deploy custom machine learning models on AWS by building a Docker image, pushing it to Amazon ECR, and running it as an inference endpoint in SageMaker. Practiced integrating and testing the endpoint via AWS Lambda.

**Key learning outcomes (brief):**
- Built a **custom Docker image** with model code using Amazon SageMaker Studio.  
- Published the image to **Amazon ECR** for versioned storage.  
- Trained a model from the custom image with sample datasets.  
- Deployed the model to an **Amazon SageMaker inference endpoint** and validated predictions.  
- Integrated endpoint invocation into AWS Lambda for application use.  
- **DIY:** Created and tested a new inference endpoint using `ml.m5.large`, updated Lambda environment variables, and validated endpoint performance.

**Tools / Services:** Amazon SageMaker Studio, Amazon ECR, AWS Lambda, Docker, Amazon S3.

**AWS SimuLearn: Introduction to Generative AI** **|** [GitHub Repo](https://tinyurl.com/john9980)

**Summary:**  
Gained hands-on experience deploying and testing foundation models on AWS. Learned to integrate Amazon SageMaker-hosted inference with AWS Lambda and validate model performance through a sample application.

**Key learning outcomes (brief):**
- Deployed a **foundation model** in Amazon SageMaker for hosted inference.  
- Updated AWS Lambda code to route requests to the SageMaker endpoint.  
- Tested and validated model outputs via a sample application.  
- **DIY:** Deployed the `huggingface-llm-falcon-7b-bf16` model, configured Lambda to use the new endpoint, and resolved resource limitations by managing active endpoints.

**AWS SimuLearn: Text-to-Image Creation Using Generative AI** **|** [GitHub Repo](https://tinyurl.com/texttoimagegenai)

**Summary:**  
Learned to deploy and integrate a text-to-image generative AI model on AWS. Used JupyterLab in Amazon SageMaker Studio to prepare, deploy, and connect the model to a practical application via AWS Lambda.

**Key learning outcomes (brief):**
- Prepared and deployed a **text-to-image** generative AI model in Amazon SageMaker.  
- Configured AWS Lambda to invoke the SageMaker endpoint for real-time image generation.  
- **DIY:** Customized model and endpoint naming, used `ml.g5.4xlarge` instance for deployment, and updated Lambda integration.

**Tools / Services:** Amazon SageMaker Studio, AWS Lambda, Generative AI Models.

**AWS SimuLearn: Image and Video Analysis** **|** [GitHub Repo](https://tinyurl.com/imagetovideoanalysis)

**Summary:**  
Gained hands-on experience in using Amazon Rekognition for automated image and video analysis. Configured AWS Lambda and Amazon S3 to trigger object detection workflows and process results.

**Key learning outcomes (brief):**
- Implemented **Amazon Rekognition label detection** to identify objects in images.  
- Integrated S3 event notifications with AWS Lambda for automated analysis.  
- **DIY:** Modified Lambda to detect a custom label ("spaceship") and validated detection results in JSON output.

**Tools / Services:** Amazon Rekognition, AWS Lambda, Amazon S3.

**AWS SimuLearn: TensorFlow and Computer Vision** **|** [GitHub Repo](https://tinyurl.com/tensorflowvision)

**Summary:**  
Applied TensorFlow with Amazon SageMaker to solve image classification problems, including sign language digit recognition. Built, trained, and deployed a CNN model to a real-time inference endpoint.

**Key learning outcomes (brief):**
- Imported and explored the **CIFAR-10 dataset** for image classification.  
- Trained a convolutional neural network (CNN) with **TensorFlow** in SageMaker.  
- **DIY:** Implemented sign language digit recognition, trained on `ml.m5.xlarge` instance for 10 epochs, and deployed the model for real-time inference.

**Tools / Services:** TensorFlow, Amazon SageMaker.

**AWS SimuLearn: Spy Drones Detection** **|** [GitHub Repo](https://tinyurl.com/spydronedetection)

**Summary:**  
Built and deployed an **XGBoost** model in Amazon SageMaker to detect spy drones based on flight data, integrating predictions via AWS Lambda.

**Key learning outcomes (brief):**
- Trained and evaluated an **XGBoost supervised learning model** using SageMaker.  
- Deployed a real-time inference endpoint and connected it to an **AWS Lambda function** for predictions.  
- **DIY:** Re-trained model with an alternate dataset (*DroneFlights.csv*) and validated detection accuracy for identifying spy drones.

**Tools / Services:** Amazon SageMaker, AWS Lambda, XGBoost, Amazon S3.

**AWS SimuLearn: Anomaly Detection** **|** [GitHub Repo](https://tinyurl.com/AnomalyDetectionAWS)

**Summary:**  
Developed and deployed a machine learning model in Amazon SageMaker to detect anomalies in datasets, integrating predictions with AWS Lambda for automated analysis.

**Key learning outcomes (brief):**
- Explored, prepared, and visualized datasets using **pandas, NumPy, and Matplotlib**.  
- Trained and evaluated an anomaly detection model in SageMaker.  
- **DIY:** Tuned hyperparameters (*50 trees*, *sample size 200*) to improve model performance, deployed a new inference endpoint, and integrated it with AWS Lambda for testing.

**Tools / Services:** Amazon SageMaker, AWS Lambda, pandas, NumPy, Matplotlib.

**AWS SimuLearn: Reinforcement Learning** **|** [GitHub Repo](https://tinyurl.com/ReinforcementLearningAWS)

**Summary:**  
Implemented a reinforcement learning model in Amazon SageMaker to train an AI agent to play tic-tac-toe, optimizing training parameters for improved performance.

**Key learning outcomes (brief):**  
- Imported and configured the training code in **SageMaker Studio**.  
- Used **SageMaker RLEstimator** to train a reinforcement learning model.  
- **DIY:** Tuned preset parameters (*improve_steps = 20,000*, *steps_between_evaluation_periods = 1,000*), retrained the model, and deployed it to an **ml.m5.xlarge** SageMaker endpoint.  
- Validated the deployed model by playing tic-tac-toe against the trained AI.

**Tools / Services:** Amazon SageMaker, SageMaker RLEstimator, AWS Lambda.

**AWS SimuLearn: Get Home Safe** **|** [GitHub Repo](https://tinyurl.com/GoHomeSafeAWS)

**Summary:**  
Built and deployed a reinforcement learning model in Amazon SageMaker to solve the MountainCar problem, optimizing hyperparameters for improved agent performance.

**Key learning outcomes (brief):**  
- Imported and configured the RL code base in **SageMaker Studio**.  
- Trained an RL model to navigate the MountainCar environment.  
- **DIY:** Adjusted hyperparameters (*epochs = 10*, *learning_rate = 0.005*), retrained the model, and deployed to an **ml.m5.large** SageMaker endpoint.  
- Tested endpoint predictions with sample data to validate model actions.

**Tools / Services:** Amazon SageMaker, RL frameworks, AWS Lambda.

**AWS SimuLearn: Customer Sentiment** **|** [GitHub Repo](https://tinyurl.com/CustomerSentimentAWS)

**Summary:**  
Implemented sentiment analysis automation using Amazon Comprehend, AWS Lambda, and Amazon S3 to process and analyze customer feedback.

**Key learning outcomes (brief):**  
- Configured **Amazon Comprehend** to analyze text sentiment from uploaded files.  
- Integrated **AWS Lambda** to call Comprehend via API and process sentiment results.  
- Set up **S3 event notifications** to trigger Lambda on file upload.  
- **DIY:** Modified Lambda to support **French** sentiment analysis, updated triggers, and validated output using a sample CSV (`sample_customer_review_fr.csv`).

**Tools / Services:** Amazon Comprehend, AWS Lambda, Amazon S3.

**AWS SimuLearn: Speech-to-Text** **|** [GitHub Repo](https://tinyurl.com/SpeechToTextAWS)

**Summary:**  
Built an automated audio transcription workflow using Amazon Transcribe, AWS Lambda, and Amazon S3 to convert uploaded audio files into text.

**Key learning outcomes (brief):**  
- Implemented **Amazon Transcribe** to convert audio recordings into text output.  
- Integrated **AWS Lambda** to trigger Transcribe via API calls.  
- Configured **Amazon S3 event notifications** to invoke Lambda when new files are uploaded.  
- **DIY:** Updated Lambda and event trigger to support **WAV** audio format, validated transcription using a sample `.wav` file.

**Tools / Services:** Amazon Transcribe, AWS Lambda, Amazon S3.

**AWS SimuLearn: Text-to-Speech** **|** [GitHub Repo](https://tinyurl.com/Text-to-SpeechAWS)

**Summary:**  
Developed an automated text-to-speech workflow using Amazon Polly, AWS Lambda, and Amazon S3 to generate spoken audio from text files.

**Key learning outcomes (brief):**  
- Implemented **Amazon Polly** to synthesize lifelike speech from text.  
- Configured **AWS Lambda** and **Amazon S3** integration to trigger speech generation automatically upon file upload.  
- **DIY:** Customized synthesis to **British English (en-GB)** using the **Amy** voice profile, validated changes in `polly_response.json`.

**Tools / Services:** Amazon Polly, AWS Lambda, Amazon S3.

**AWS SimuLearn: Extract Text from Docs** **|** [GitHub Repo](https://tinyurl.com/Extract-TextfromDocsAWS)

**Summary:**  
Built a document-processing workflow that uses Amazon Textract, AWS Lambda, and Amazon S3 to automatically extract structured data from uploaded images.

**Key learning outcomes (brief):**  
- Implemented **Amazon Textract** to extract **form data** from scanned documents.  
- Configured **AWS Lambda** and **Amazon S3** event triggers for automated extraction upon file upload.  
- Reviewed extracted data through **Amazon CloudWatch Logs**.  
- **DIY:** Enhanced extraction to include **table data**, validating results via `textract_response.json` and CloudWatch logs.

**Tools / Services:** Amazon Textract, AWS Lambda, Amazon S3, Amazon CloudWatch.

**AWS SimuLearn: Data Ingestion Methods** **|** [GitHub Repo](https://tinyurl.com/DataIngestionMethodsAWS)

**Summary:**  
Built a real-time and batch data ingestion pipeline leveraging Amazon Kinesis Data Firehose, AWS Lambda, Amazon S3, and analytics services to process and query clickstream data.

**Key learning outcomes (brief):**  
- Created an **Amazon Kinesis Data Firehose** stream to ingest and store clickstream data in **Amazon S3**.  
- Implemented **AWS Lambda** transformations for incoming data.  
- Built real-time querying capability with **AWS Glue** and **Amazon Athena**.  
- **DIY:** Enhanced the Firehose pipeline to send **real-time analytics to Amazon DynamoDB** and optimized ingestion latency by reducing **S3 buffer interval** from 300s to 60s.

**Tools / Services:** Amazon Kinesis Data Firehose, AWS Lambda, Amazon S3, Amazon DynamoDB, AWS Glue, Amazon Athena.

**AWS SimuLearn: Computing Solutions** **|** [GitHub Repo](https://tinyurl.com/ComputingSolutionsAWS)

**Summary:**  
Gained hands-on experience managing Amazon EC2 instances, exploring compute options, and modifying configurations for different workloads.

**Key learning outcomes (brief):**  
- Explored **Amazon EC2 instance types** and filtered based on attributes.  
- Connected to instances using **EC2 Instance Connect** and **Session Manager**.  
- Retrieved EC2 instance metadata using the public IP address.  
- Managed instance lifecycle (start/stop) via the **Amazon EC2 console**.  
- **DIY:** Reconfigured the instance type to a **larger general-purpose m4.large** instance for improved capacity.

**Tools / Services:** Amazon EC2, EC2 Instance Connect, AWS Systems Manager Session Manager.

**AWS SimuLearn: Cloud Computing Essentials** **|** [GitHub Repo](https://tinyurl.com/john00777)

**Summary:**  
Gained practical cloud fundamentals and hands-on experience migrating a static website to Amazon S3. Learned how to configure S3 static website hosting, secure buckets with policies and block-public-access settings, use server-side encryption, and understand Regions, availability, and S3 endpoints.

**Key learning outcomes (brief):**
- Enable and configure **S3 static website hosting** (index & error documents, website endpoint).  
- Review and apply **bucket policies** and public-access settings to control read access.  
- Understand S3 **storage classes**, Regions, and how S3 provides high availability and scalability.  
- Configure **server-side encryption** for S3 objects.  
- Performed lab tasks: inspected bucket contents, uploaded objects, and validated website endpoint.  

**AWS SimuLearn: Cloud First Steps** **|** [GitHub Repo](https://tinyurl.com/CloudFirstStepsAWS)

**Summary:**  
Built foundational AWS skills by deploying and configuring EC2 instances, introducing the basics of cloud compute provisioning and automation.

**Key learning outcomes (brief):**  
- Launched an **Amazon EC2 instance** and configured a **user data script** to automatically display instance details in a browser.  
- Understood the relationship between **EC2 instances, user data automation, and availability zones**.  
- **DIY:** Deployed a **second EC2 instance in a different Availability Zone** within the same AWS Region for high availability.

**Tools / Services:** Amazon EC2, AWS Management Console.

**AWS SimuLearn: Serverless Foundations** **|** [GitHub Repo](https://tinyurl.com/ServerlessFoundationsAWS)

**Summary:**  
Introduced to AWS serverless computing by building and deploying Python-based Lambda functions, focusing on event-driven execution and lightweight processing.

**Key learning outcomes (brief):**  
- Created, deployed, and tested an **AWS Lambda function** using **Python**.  
- Learned how **serverless functions** scale automatically and execute on demand without server provisioning.  
- **DIY:** Enhanced Lambda code to dynamically return a **sentiment label (positive, neutral, negative)** based on the `emoji_type` value in incoming JSON payloads.

**Tools / Services:** AWS Lambda, Python, AWS Management Console.

**AWS SimuLearn: Networking Concepts** **|** [GitHub Repo](https://tinyurl.com/NetworkingConceptsAWS)

**Summary:**  
Explored foundational **AWS networking** concepts by working with VPC components, route tables, internet gateways, and security groups to control traffic flow and connectivity.

**Key learning outcomes (brief):**  
- Gained hands-on experience with **Virtual Private Cloud (VPC)** structure and components.  
- Configured **route tables** to direct internet-bound traffic through an **internet gateway**.  
- Managed **security group rules** to control inbound network access.  
- **DIY:** Enabled database access by allowing TCP traffic over **port 3306** from a web server subnet to a DB server subnet.

**Tools / Services:** Amazon VPC, Route Tables, Internet Gateway, Security Groups.

**AWS SimuLearn: Core Security Concepts** **|** [GitHub Repo](https://tinyurl.com/CoreSecurityConceptsAWS)

**Summary:**  
Developed core AWS security administration skills by creating and managing **IAM groups and users**, and assigning appropriate AWS-managed policies for secure, role-based access control.

**Key learning outcomes (brief):**  
- Created **IAM groups** and added users for structured access management.  
- Attached **AWS managed policies** to groups to enforce least privilege principles.  
- **DIY:** Granted the `SupportEngineers` group **read-only access** to both **Amazon EC2** and **Amazon RDS** by applying `EC2ReadOnlyAccess` and `RDSReadOnlyAccess` managed policies.

**Tools / Services:** AWS Identity and Access Management (IAM), Amazon RDS, Amazon EC2.

**AWS SimuLearn: Databases in Practice** **|** [GitHub Repo](https://tinyurl.com/DatabaseInPracticeAWS)

**Summary:**  
Gained practical experience with **AWS managed database services**, focusing on deploying and configuring **Amazon RDS** for scalability, availability, and disaster recovery.

**Key learning outcomes (brief):**  
- Explored **AWS database offerings** and their use cases.  
- Launched and configured an **Amazon RDS instance**.  
- Enabled **Multi-AZ deployment** for high availability.  
- Configured **automated backups** for disaster recovery.  
- **DIY:** Created a **read replica** (`my-database-read-replica`) of the primary RDS instance (`my-database`) using a **db.t3.xlarge** configuration to offload read traffic.

**Tools / Services:** Amazon RDS, Multi-AZ, Read Replicas, Automated Backups.

**AWS SimuLearn: Cloud Economics** **|** [GitHub Repo](https://tinyurl.com/CloudEconomicsAWS)

**Summary:**  
Learned to apply **AWS cost optimization principles** by using AWS Pricing Calculator to forecast and manage infrastructure costs.

**Key learning outcomes (brief):**  
- Created **logical pricing groups** for organized cost estimation.  
- Built an **Amazon EC2 usage estimate** based on workload requirements.  
- **DIY:** Rightsized EC2 instances from **t3** to **t2.micro** in the cost estimate, then generated a **new shareable price estimate URL**.

**Tools / Services:** AWS Pricing Calculator, EC2 Cost Estimation, Rightsizing.














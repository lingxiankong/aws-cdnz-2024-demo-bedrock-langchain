import os.path
import json

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_kms as kms,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_opensearchserverless as opensearch,
    aws_lambda as lambda_,
    aws_lambda_python_alpha as lambda_python,
    aws_bedrock as bedrock,
)
from constructs import Construct

DATA_SOURCE_S3_PREFIX = "knowledgebase_data_source"
BEDROCK_AGENT_FM = "anthropic.claude-3-sonnet-20240229-v1:0"
# Embedding model would affect the dimensions, see https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-setup.html
BEDROCK_EMBEDDING_MODEL = bedrock.FoundationModelIdentifier.COHERE_EMBED_ENGLISH_V3


class BedrockAgentStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 data source
        s3_kms_key = self.create_s3_kms_key()
        agent_datasource_bucket = self.create_data_source_bucket(s3_kms_key)
        self.upload_files_to_s3(agent_datasource_bucket)

        # Agent IAM role
        agent_role = self.create_agent_execution_role(agent_datasource_bucket)

        # Create OpenSearch collection and the relevant resources using Lambda
        opensearch_layer = self.create_lambda_layer("opensearch_layer")
        cfn_collection, vector_field_name, vector_index_name, lambda_cr = self.create_opensearch_index(
            agent_role, opensearch_layer
        )

        # Create knowledge base
        knowledge_base = self.create_knowledgebase(
            vector_field_name,
            vector_index_name,
            cfn_collection,
            agent_role,
            lambda_cr,
        )
        self.create_agent_data_source(knowledge_base, agent_datasource_bucket)

        # Create agent
        agent = self.create_bedrock_agent(agent_role, knowledge_base)
        cdk.CfnOutput(self, "AgentID", value=agent.attr_agent_id)

    def create_agent_execution_role(self, bucket):
        agent_role = iam.Role(
            self,
            "BedrockAgentRole",
            role_name="AmazonBedrockExecutionRoleForAgents_test",  # must be AmazonBedrockExecutionRoleForAgents_string
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
        )
        policy_statements = [
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["bedrock:InvokeModel"],
                resources=[f"arn:aws:bedrock:{cdk.Aws.REGION}::foundation-model/*"],
            ),
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["s3:GetObject", "s3:ListBucket"],
                resources=[
                    f"arn:aws:s3:::{bucket.bucket_name}",
                    f"arn:aws:s3:::{bucket.bucket_name}/*",
                ],
                conditions={
                    "StringEquals": {
                        "aws:ResourceAccount": f"{cdk.Aws.ACCOUNT_ID}",
                    },
                },
            ),
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["bedrock:Retrieve", "bedrock:RetrieveAndGenerate"],
                resources=[f"arn:aws:bedrock:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:knowledge-base/*"],
                conditions={
                    "StringEquals": {
                        "aws:ResourceAccount": f"{cdk.Aws.ACCOUNT_ID}",
                    },
                },
            ),
        ]

        for statement in policy_statements:
            agent_role.add_to_policy(statement)

        return agent_role

    def create_s3_kms_key(self):
        kms_key = kms.Key(
            self,
            "S3Key",
            alias=f"alias/{cdk.Aws.STACK_NAME}/s3_key",
            enable_key_rotation=True,
            pending_window=cdk.Duration.days(7),
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        kms_key.grant_encrypt_decrypt(
            iam.AnyPrincipal().with_conditions(
                {
                    "StringEquals": {
                        "kms:CallerAccount": f"{cdk.Aws.ACCOUNT_ID}",
                        "kms:ViaService": f"s3.{cdk.Aws.REGION}.amazonaws.com",
                    },
                }
            )
        )

        kms_key.grant_encrypt_decrypt(iam.ServicePrincipal(f"logs.{cdk.Aws.REGION}.amazonaws.com"))

        return kms_key

    def create_data_source_bucket(self, kms_key):
        agent_datasource_bucket = s3.Bucket(
            self,
            "AgentDataSource",
            bucket_name=f"agent-datasource",
            versioned=False,
            auto_delete_objects=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=kms_key,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
        )
        cdk.CfnOutput(self, "DataSourceBucketName", value=agent_datasource_bucket.bucket_name)

        return agent_datasource_bucket

    def upload_files_to_s3(self, bucket):
        s3deploy.BucketDeployment(
            self,
            "KnowledgeBaseDocumentDeployment",
            sources=[
                s3deploy.Source.asset(
                    os.path.join(
                        os.getcwd(),
                        "assets",
                        f"{DATA_SOURCE_S3_PREFIX}/ec2.zip",
                    )
                )
            ],
            destination_bucket=bucket,
            destination_key_prefix=DATA_SOURCE_S3_PREFIX,
            retain_on_delete=False,
        )

        return

    def create_lambda_layer(self, layer_name):
        """
        create a Lambda layer with necessary dependencies.
        """
        # Create the Lambda layer
        layer = lambda_python.PythonLayerVersion(
            self,
            layer_name,
            entry=os.path.join(os.getcwd(), "code/layers", layer_name),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            compatible_architectures=[lambda_.Architecture.ARM_64],
            description="A layer for new version of python package",
            layer_version_name=layer_name,
        )

        return layer

    def create_opensearch_index(self, agent_role, opensearch_layer):

        vector_index_name = "bedrock-knowledgebase-index"
        vector_field_name = "bedrock-knowledgebase-default-vector"

        agent_role_arn = agent_role.role_arn

        create_index_lambda_execution_role = iam.Role(
            self,
            "CreateIndexExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Role for OpenSearch access",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ],
        )
        create_index_lambda_execution_role_arn = create_index_lambda_execution_role.role_arn

        cfn_collection = opensearch.CfnCollection(
            self,
            "BedrockAgentTest",
            name=f"bedrock-agent-test",
            description="Test Bedrock Agent",
            type="VECTORSEARCH",
        )
        cfn_collection_name = cfn_collection.name

        opensearch_policy_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=["aoss:APIAccessAll"],
            resources=[f"arn:aws:aoss:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:collection/{cfn_collection.attr_id}"],
        )

        # Attach the custom policy to the roles
        create_index_lambda_execution_role.add_to_policy(opensearch_policy_statement)
        agent_role.add_to_policy(opensearch_policy_statement)

        policy_json = {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{cfn_collection_name}"],
                }
            ],
            "AWSOwnedKey": True,
        }

        encryption_policy = cdk.CfnResource(
            self,
            "EncryptionPolicy",
            type="AWS::OpenSearchServerless::SecurityPolicy",
            properties={
                "Name": "chatbot-index-encryption-policy",
                "Type": "encryption",
                "Description": "Encryption policy for Bedrock collection.",
                "Policy": json.dumps(policy_json),
            },
        )

        policy_json = [
            {
                "Rules": [
                    {
                        "ResourceType": "collection",
                        "Resource": [f"collection/{cfn_collection_name}"],
                    },
                    {
                        "ResourceType": "dashboard",
                        "Resource": [f"collection/{cfn_collection_name}"],
                    },
                ],
                "AllowFromPublic": True,
            }
        ]

        network_policy = cdk.CfnResource(
            self,
            "NetworkPolicy",
            type="AWS::OpenSearchServerless::SecurityPolicy",
            properties={
                "Name": "chatbot-index-network-policy",
                "Type": "network",
                "Description": "Network policy for Bedrock collection",
                "Policy": json.dumps(policy_json),
            },
        )

        policy_json = [
            {
                "Description": "Access for cfn user",
                "Rules": [
                    {
                        "ResourceType": "index",
                        "Resource": ["index/*/*"],
                        "Permission": ["aoss:*"],
                    },
                    {
                        "ResourceType": "collection",
                        "Resource": [f"collection/{cfn_collection_name}"],
                        "Permission": ["aoss:*"],
                    },
                ],
                "Principal": [
                    agent_role_arn,
                    create_index_lambda_execution_role_arn,
                ],
            }
        ]

        data_policy = cdk.CfnResource(
            self,
            "DataPolicy",
            type="AWS::OpenSearchServerless::AccessPolicy",
            properties={
                "Name": "chatbot-index-data-policy",
                "Type": "data",
                "Description": "Data policy for Bedrock collection.",
                "Policy": json.dumps(policy_json),
            },
        )

        cfn_collection.add_dependency(network_policy)
        cfn_collection.add_dependency(encryption_policy)
        cfn_collection.add_dependency(data_policy)

        create_index_lambda = lambda_.Function(
            self,
            "CreateOpenSearchIndex",
            function_name=f"{cdk.Aws.STACK_NAME}-create-index",
            runtime=lambda_.Runtime.PYTHON_3_12,  # Runtime.FROM_IMAGE if using container image
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset(os.path.join(os.getcwd(), "code/lambdas", "create-index-lambda")),
            layers=[opensearch_layer],
            environment={
                "REGION_NAME": cdk.Aws.REGION,
                "COLLECTION_HOST": cfn_collection.attr_collection_endpoint,
                "VECTOR_INDEX_NAME": vector_index_name,
                "VECTOR_FIELD_NAME": vector_field_name,
            },
            role=create_index_lambda_execution_role,
            timeout=cdk.Duration.minutes(15),
        )

        lambda_provider = cdk.custom_resources.Provider(
            self,
            "LambdaCreateIndexCustomProvider",
            on_event_handler=create_index_lambda,
        )

        lambda_cr = cdk.CustomResource(
            self,
            "LambdaCreateIndexCustomResource",
            service_token=lambda_provider.service_token,
        )

        return (
            cfn_collection,
            vector_field_name,
            vector_index_name,
            lambda_cr,
        )

    def create_knowledgebase(
        self,
        vector_field_name,
        vector_index_name,
        cfn_collection,
        agent_role,
        lambda_cr,
    ):

        kb_name = "BedrockKnowledgeBase"
        text_field = "AMAZON_BEDROCK_TEXT_CHUNK"
        metadata_field = "AMAZON_BEDROCK_METADATA"
        agent_role_arn = agent_role.role_arn

        embed_model = bedrock.FoundationModel.from_foundation_model_id(
            self,
            "embedding_model",
            BEDROCK_EMBEDDING_MODEL,
        )

        cfn_knowledge_base = bedrock.CfnKnowledgeBase(
            self,
            "BedrockOpenSearchKnowledgeBase",
            knowledge_base_configuration=bedrock.CfnKnowledgeBase.KnowledgeBaseConfigurationProperty(
                type="VECTOR",
                vector_knowledge_base_configuration=bedrock.CfnKnowledgeBase.VectorKnowledgeBaseConfigurationProperty(
                    embedding_model_arn=embed_model.model_arn
                ),
            ),
            name=kb_name,
            role_arn=agent_role_arn,
            storage_configuration=bedrock.CfnKnowledgeBase.StorageConfigurationProperty(
                type="OPENSEARCH_SERVERLESS",
                opensearch_serverless_configuration=bedrock.CfnKnowledgeBase.OpenSearchServerlessConfigurationProperty(
                    collection_arn=cfn_collection.attr_arn,
                    field_mapping=bedrock.CfnKnowledgeBase.OpenSearchServerlessFieldMappingProperty(
                        metadata_field=metadata_field,
                        text_field=text_field,
                        vector_field=vector_field_name,
                    ),
                    vector_index_name=vector_index_name,
                ),
            ),
            description="Use this for returning descriptive answers and instructions directly from AWS EC2 Documentation. Use to answer qualitative/guidance questions such as 'how do I',  general instructions and guidelines.",
        )

        child = None
        for child in lambda_cr.node.children:
            if isinstance(child, cdk.CustomResource):
                break

        if child:
            cfn_knowledge_base.add_dependency(child)

        return cfn_knowledge_base

    def create_agent_data_source(self, knowledge_base, bucket):
        data_source_bucket_arn = f"arn:aws:s3:::{bucket.bucket_name}"

        cfn_data_source = bedrock.CfnDataSource(
            self,
            "BedrockKnowledgeBaseDataSource",
            name="BedrockKnowledgeBaseSource",
            data_source_configuration=bedrock.CfnDataSource.DataSourceConfigurationProperty(
                type="S3",
                s3_configuration=bedrock.CfnDataSource.S3DataSourceConfigurationProperty(
                    bucket_arn=data_source_bucket_arn,
                    # the properties below are optional
                    bucket_owner_account_id=cdk.Aws.ACCOUNT_ID,
                    inclusion_prefixes=[f"{DATA_SOURCE_S3_PREFIX}/"],
                ),
            ),
            knowledge_base_id=knowledge_base.attr_knowledge_base_id,
            data_deletion_policy="RETAIN",
        )

        return cfn_data_source

    def create_bedrock_agent(
        self,
        agent_role,
        cfn_knowledge_base,
    ):
        agent_role_arn = agent_role.role_arn

        agent_instruction = (
            "You are a friendly chat bot. You have access to a knowledge base to answer questions AWS "
            "related questions."
        )

        cfn_agent = bedrock.CfnAgent(
            self,
            "TestBedrockAgent",
            agent_name="TestBedrockAgent",
            agent_resource_role_arn=agent_role_arn,
            description="Bedrock Test Agent",
            foundation_model=BEDROCK_AGENT_FM,
            idle_session_ttl_in_seconds=3600,
            instruction=agent_instruction,
            knowledge_bases=[
                bedrock.CfnAgent.AgentKnowledgeBaseProperty(
                    description=cfn_knowledge_base.description,
                    knowledge_base_id=cfn_knowledge_base.attr_knowledge_base_id,
                )
            ],
            # https://docs.aws.amazon.com/bedrock/latest/userguide/advanced-prompts-configure.html
            # prompt_override_configuration=bedrock.CfnAgent.PromptOverrideConfigurationProperty(
            #     prompt_configurations=[
            #         bedrock.CfnAgent.PromptConfigurationProperty(
            #             base_prompt_template=PREPROCESSING_TEMPLATE,
            #             inference_configuration=bedrock.CfnAgent.InferenceConfigurationProperty(
            #                 maximum_length=256,
            #                 stop_sequences=["\n\nHuman:"],
            #                 temperature=0,
            #                 top_k=250,
            #                 top_p=1,
            #             ),
            #             parser_mode="DEFAULT",
            #             prompt_creation_mode="OVERRIDDEN",
            #             prompt_state="ENABLED",
            #             prompt_type="PRE_PROCESSING",
            #         ),
            #         bedrock.CfnAgent.PromptConfigurationProperty(
            #             base_prompt_template=ORCHESTRATION_TEMPLATE,
            #             inference_configuration=bedrock.CfnAgent.InferenceConfigurationProperty(
            #                 maximum_length=2048,
            #                 stop_sequences=["</function_call>", "</answer>", "/error"],
            #                 temperature=0,
            #                 top_k=250,
            #                 top_p=1,
            #             ),
            #             parser_mode="DEFAULT",
            #             prompt_creation_mode="OVERRIDDEN",
            #             prompt_state="ENABLED",
            #             prompt_type="ORCHESTRATION",
            #         ),
            #         bedrock.CfnAgent.PromptConfigurationProperty(
            #             base_prompt_template=ORCHESTRATION_TEMPLATE,
            #             inference_configuration=bedrock.CfnAgent.InferenceConfigurationProperty(
            #                 maximum_length=2048,
            #                 stop_sequences=["\n\nHuman:"],
            #                 temperature=0,
            #                 top_k=250,
            #                 top_p=1,
            #             ),
            #             parser_mode="DEFAULT",
            #             prompt_creation_mode="OVERRIDDEN",
            #             prompt_state="ENABLED",
            #             prompt_type="KNOWLEDGE_BASE_RESPONSE_GENERATION",
            #         ),
            #     ]
            # ),
        )

        return cfn_agent

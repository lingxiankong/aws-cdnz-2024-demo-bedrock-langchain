#!/usr/bin/env python3
import os

import aws_cdk as cdk

from bedrock_agent.bedrock_agent_stack import BedrockAgentStack


app = cdk.App()
BedrockAgentStack(
    app,
    "LingxianTestBedrockAgent",
    env=cdk.Environment(account=os.getenv("CDK_DEFAULT_ACCOUNT"), region=os.getenv("CDK_DEFAULT_REGION")),
    synthesizer=cdk.DefaultStackSynthesizer(qualifier="lingxian")
)

app.synth()

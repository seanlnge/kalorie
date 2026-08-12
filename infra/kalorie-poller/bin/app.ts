#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { KaloriePollerStack } from "../lib/kalorie-poller-stack";

const app = new cdk.App();

new KaloriePollerStack(app, "KaloriePollerStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? "us-east-1",
  },
  description: "6h Kalshi + kalorie-v6 snapshot writer to S3",
});

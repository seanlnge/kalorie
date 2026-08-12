import * as fs from "node:fs";
import * as path from "node:path";
import * as cdk from "aws-cdk-lib";
import { Duration } from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { DockerImageCode, DockerImageFunction } from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as scheduler from "aws-cdk-lib/aws-scheduler";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import { Construct } from "constructs";

function findWorkspaceRoot(startDir: string): string {
  let current = startDir;
  for (let i = 0; i < 10; i += 1) {
    const dockerfile = path.join(current, "kalorie2", "lambda_poller", "Dockerfile");
    const modelDir = path.join(current, "models", "kalorie-v6", "artifacts", "model.json");
    if (fs.existsSync(dockerfile) && fs.existsSync(modelDir)) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }
  throw new Error(
    "Could not find workspace root containing kalorie2/lambda_poller/Dockerfile and models/kalorie-v6",
  );
}

export class KaloriePollerStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Public read + CORS so kalorie-desk can fetch latest.json from the browser.
    // Objects are model snapshots only (no secrets).
    const snapshotBucket = new s3.Bucket(this, "SnapshotBucket", {
      blockPublicAccess: new s3.BlockPublicAccess({
        blockPublicAcls: true,
        blockPublicPolicy: false,
        ignorePublicAcls: true,
        restrictPublicBuckets: false,
      }),
      publicReadAccess: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      versioned: true,
      cors: [
        {
          allowedMethods: [s3.HttpMethods.GET, s3.HttpMethods.HEAD],
          allowedOrigins: ["*"],
          allowedHeaders: ["*"],
          maxAge: 300,
        },
      ],
      lifecycleRules: [
        {
          id: "expire-old-snapshots",
          expiration: Duration.days(90),
          noncurrentVersionExpiration: Duration.days(30),
        },
      ],
    });

    new cdk.CfnOutput(this, "SnapshotPublicBaseUrl", {
      value: `https://${snapshotBucket.bucketRegionalDomainName}`,
    });

    // Paste OPENAI_API_KEY after deploy (see README):
    //   aws secretsmanager put-secret-value \
    //     --secret-id <OpenAiSecretArn> \
    //     --secret-string '{"OPENAI_API_KEY":"sk-..."}'
    const openAiSecret = new secretsmanager.Secret(this, "OpenAiSecret", {
      description: "Kalorie poller OpenAI credentials (JSON: OPENAI_API_KEY)",
      secretStringValue: cdk.SecretValue.unsafePlainText(
        JSON.stringify({ OPENAI_API_KEY: "REPLACE_ME" }),
      ),
    });

    const workspaceRoot = findWorkspaceRoot(__dirname);

    const pollerLogGroup = new logs.LogGroup(this, "SnapshotPollerLogGroup", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const pollerFn = new DockerImageFunction(this, "SnapshotPoller", {
      code: DockerImageCode.fromImageAsset(workspaceRoot, {
        file: "kalorie2/lambda_poller/Dockerfile",
        exclude: [
          "**/node_modules",
          "**/.git",
          "**/__pycache__",
          "kalorie2/artifacts",
          "kalorie2/web/node_modules",
          "kalorie2/web/dist",
          "**/cdk.out",
        ],
      }),
      memorySize: 1024,
      timeout: Duration.minutes(15),
      architecture: lambda.Architecture.X86_64,
      environment: {
        SNAPSHOT_BUCKET: snapshotBucket.bucketName,
        MODEL_NAME: "kalorie-v6",
        MODELS_ROOT: "/opt/models",
        LIVE_WEB_EVIDENCE: "true",
        WEB_SEARCH_MODEL: "gpt-5.4-mini",
        WEB_EVIDENCE_MAX_WORKERS: "12",
        KALSHI_SCAN_ALL_OPEN_MARKETS: "false",
        OPENAI_SECRET_ARN: openAiSecret.secretArn,
      },
      logGroup: pollerLogGroup,
    });

    snapshotBucket.grantPut(pollerFn);
    openAiSecret.grantRead(pollerFn);

    // EventBridge Scheduler with America/New_York so 5am / 8pm Eastern
    // stay correct across EST/EDT (classic EventBridge Rules are UTC-only).
    const schedulerRole = new iam.Role(this, "PollerSchedulerRole", {
      assumedBy: new iam.ServicePrincipal("scheduler.amazonaws.com"),
    });
    pollerFn.grantInvoke(schedulerRole);

    const easternSchedules: { id: string; hour: string; label: string }[] = [
      { id: "FiveAmEastern", hour: "5", label: "5:00 AM Eastern" },
      { id: "EightPmEastern", hour: "20", label: "8:00 PM Eastern" },
    ];
    for (const item of easternSchedules) {
      new scheduler.CfnSchedule(this, item.id, {
        description: `Run Kalorie snapshot poller at ${item.label}`,
        flexibleTimeWindow: { mode: "OFF" },
        scheduleExpression: `cron(0 ${item.hour} * * ? *)`,
        scheduleExpressionTimezone: "America/New_York",
        target: {
          arn: pollerFn.functionArn,
          roleArn: schedulerRole.roleArn,
        },
      });
    }

    new cdk.CfnOutput(this, "SnapshotBucketName", {
      value: snapshotBucket.bucketName,
    });
    new cdk.CfnOutput(this, "PollerFunctionName", {
      value: pollerFn.functionName,
    });
    new cdk.CfnOutput(this, "OpenAiSecretArn", {
      value: openAiSecret.secretArn,
    });
    new cdk.CfnOutput(this, "ScheduleTimezone", {
      value: "America/New_York",
    });
    new cdk.CfnOutput(this, "ScheduleHoursEastern", {
      value: "05:00,20:00",
    });
  }
}

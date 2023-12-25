import boto3, botocore

def check_login():
    sts = boto3.client('sts')
    try:
        sts.get_caller_identity()
        return True
    except botocore.exceptions.UnauthorizedSSOTokenError:
        print('We zijn niet geautoriseerd op AWS. Kijk of systeem wel goed geconfigureerd is.')
        return False
    except botocore.exceptions.SSOTokenLoadError:
        print('AWS SSO Token is niet actief. Graag aws sso login --profile \'AWSIntroTraining-408122842185\' uitvoeren.')
        return False
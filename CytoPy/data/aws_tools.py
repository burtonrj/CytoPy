#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module contains methods for accessing AWS S3 buckets for data storage.
Before using CytoPy with AWS, an AWS account is required and configuration
performed, see: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import boto3


def create_bucket(bucket_name: str,
                  region: str or None = None) -> None:
    """
    Generate a new bucket. If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    Parameters
    ----------
    bucket_name: str
    region: str, optional

    Returns
    -------
    None
    """
    if region is None:
        s3_client = boto3.client('s3')
        s3_client.create_bucket(Bucket=bucket_name)
    else:
        s3_client = boto3.client('s3', region_name=region)
        location = {'LocationConstraint': region}
        s3_client.create_bucket(Bucket=bucket_name,
                                CreateBucketConfiguration=location)


def list_available_buckets() -> list:
    """
    Lists available buckets in configured AWS account

    Returns
    -------
    List
    """
    s3 = boto3.client('s3')
    return list(s3.list_buckets())

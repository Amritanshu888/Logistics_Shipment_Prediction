#!bin/bash

az group create -n shippingapptfstate -l eastus2
 
az storage account create -n shippingapptfstate -g shippingapptfstate -l eastus2 --sku "Standard_LRS"
 
az storage container create -n tfstate --account-name shippingapptfstate
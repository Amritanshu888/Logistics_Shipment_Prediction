terraform {
  backend "azurerm" {
    resource_group_name  = "shippingapptfstate"
    storage_account_name = "shippingapptfstate"
    container_name       = "tfstate"
    key                  = "prod.terraform.tfstate"
  }
}

provider "azurerm" {
  features {}
}

module "web_app" {
  source = "./web_app"
}

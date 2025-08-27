# Create the database
Create database Nepa_Wholesale ;

#Use database
Use Nepa_Wholesale ;

# Create Table for Disposable Category
Create table Disposable_category(
	Product_ID int NOT NULL,
    Display_Name varchar(225),
    Barcode VARCHAR(50),
    Product_Sub_Category varchar(80),
    Product_Category varchar(80),
    Brand varchar(80),
    Puff_count int ,
    Flavor varchar(80),
    Nicotine_strength VARCHAR(20) ,
    Pack_count varchar(30),
    primary key(Product_ID)
);

#Create table for Tobaccos_Category
Create table Tobaccos_category(
		Product_ID int not null,
        Display_Name varchar(225),
        Barcode varchar(50),
        Product_Sub_Category varchar(80),
        Product_Category varchar(80),
        Brand varchar(80),
        Flavor varchar(80),
        Quantity varchar(80),
        Other_Features varchar(200),
        
        primary key (Product_ID)
);

#Create table for Cigars_Category
Create table Cigars_category(
	Product_ID int not null,
    Display_Name varchar(225),
    Barcode varchar(50),
    Product_Sub_Category varchar(80),
    Product_Category varchar(80),
    Brand varchar(80),
    Flavor varchar(80),
    Packet_count varchar(30),
    Other_Features varchar(200),
    
    primary key (Product_ID)
);


# Insert the data into the table

select count(*) from tobaccos_category;
select count(*) from cigars_category;
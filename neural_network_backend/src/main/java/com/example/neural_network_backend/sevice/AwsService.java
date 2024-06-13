package com.example.neural_network_backend.sevice;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.mongodb.core.aggregation.ArithmeticOperators.Log;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import com.amazonaws.services.s3.model.S3Object;
import com.amazonaws.services.s3.model.S3ObjectInputStream;
import com.amazonaws.util.IOUtils;
import com.example.neural_network_backend.model.FileNamesDTO;
import com.example.neural_network_backend.model.UserDTO;
import com.example.neural_network_backend.repository.FileNamesRepository;
import com.example.neural_network_backend.repository.UserRepository;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.model.PutObjectRequest;

import java.io.File;

@Service
public class AwsService {

    @Value("${spring.application.bucket.name}")
    private String bucketName;

    @Autowired
    private AmazonS3 s3Client;

    @Autowired 
    UserRepository userRepo;

    @Autowired
    FileNamesRepository fileNamesRepo;


    // so this method saves the file to our amzaon s3 storage 
    public String uploadPhoto(String userId, MultipartFile file){

        Optional<UserDTO> user =  userRepo.findByEmail(userId);

        // we convert the MultipartFile to a file 
        File fileObject = convertMultiPartFileToFile(file);

        // we create a file name so we get the time and the file name which we get from the frontend 
        String fileName = "photo"+user.get().getId()+user.get().getEmail()+System.currentTimeMillis()+"_"+file.getOriginalFilename();

        user.get().setPhotoId(fileName);
        if(!user.get().getPhotoId().equals("")){
            s3Client.deleteObject(bucketName, user.get().getPhotoId());
        }
        userRepo.save(user.get());


        // next we are using the Amazons3 object which we Autowired 
        // in this we need to crate a new putObjectRequest which takes the bucketname which we get from our proties file 
        // and than the filename and the file itself 
        s3Client.putObject(new PutObjectRequest(bucketName, fileName, fileObject));
        //next we need to delete the fileobject 
        fileObject.delete();
        return "Success: File " + fileName + " was uploaded. ";
    }


    // so this method saves the file to our amzaon s3 storage 
    public String uploadFile(String userId, MultipartFile file){

        Optional<UserDTO> user =  userRepo.findByEmail(userId);

        // we convert the MultipartFile to a file 
        File fileObject = convertMultiPartFileToFile(file);

        // we create a file name so we get the time and the file name which we get from the frontend 
        String fileName =user.get().getEmail()+System.currentTimeMillis()+"_"+file.getOriginalFilename();

        // next we are using the Amazons3 object which we Autowired 
        // in this we need to crate a new putObjectRequest which takes the bucketname which we get from our proties file 
        // and than the filename and the file itself 
        s3Client.putObject(new PutObjectRequest(bucketName, fileName, fileObject));


        // this is putting the name inside the fileName 
        

        FileNamesDTO addFileName = new FileNamesDTO();
        addFileName.setUserId(user.get().getId());
        addFileName.setFileName(fileName);
        fileNamesRepo.save(addFileName);

        //next we need to delete the fileobject 
        fileObject.delete();
        return "Success: File " + fileName + " was uploaded. ";
    }

    


    // this one download the file from our s3 storage 
    public byte[] downloadFile(String fileName){
        // its kinda like sql we need to input our buckname than our filename 

        if(fileName == null){
            
            S3Object s3Object = s3Client.getObject(bucketName, "Screenshot 2024-05-23 at 4.22.44 PM.png");
            S3ObjectInputStream inputStream = s3Object.getObjectContent();
            try {
            
                byte[] content = IOUtils.toByteArray(inputStream);
                return content;
            } catch (Exception e) {
                // TODO: handle exception
            }
            return null;
            
            
        }else{
            S3Object s3Object = s3Client.getObject(bucketName, fileName);
        

            S3ObjectInputStream inputStream = s3Object.getObjectContent();

            try {
                
                byte[] content = IOUtils.toByteArray(inputStream);
                return content;
            } catch (Exception e) {
                // TODO: handle exception
            }

            return null;
        }
        
        

    }

    // this method is used to delete files from our s3 storage 
    public String deleteFile(String fileName){
        s3Client.deleteObject(bucketName, fileName);
        return "Delete Success: File " + fileName + " was uploaded. ";
    }

    // this is a helper method for coverting a MultipartFile to a File
    private File convertMultiPartFileToFile(MultipartFile file) {
        File convertedFile = new File(file.getOriginalFilename());
        try (FileOutputStream fos = new FileOutputStream(convertedFile)) {
            fos.write(file.getBytes());
        } catch (IOException e) {

        }
        return convertedFile;
    }
}


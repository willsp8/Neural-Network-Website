package com.example.neural_network_backend.security;

import java.io.IOException;
import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken;
import org.springframework.security.oauth2.core.user.DefaultOAuth2User;
import org.springframework.security.web.authentication.SavedRequestAwareAuthenticationSuccessHandler;
import org.springframework.stereotype.Component;

import com.example.neural_network_backend.model.RegistrationSource;
import com.example.neural_network_backend.model.UserDTO;
import com.example.neural_network_backend.model.UserRole;
import com.example.neural_network_backend.repository.UserRepository;

import org.springframework.security.core.Authentication;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import lombok.RequiredArgsConstructor;

import java.util.List;
import java.util.Map;


@Component
@RequiredArgsConstructor
public class OAuth2LoginSuccessHandler extends SavedRequestAwareAuthenticationSuccessHandler {
    
    private final UserRepository userRepo;
   
    @Value("frontend.url")
    private String frontendUrl;

    @Override
    public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response, Authentication authentication) throws ServletException, IOException {

        OAuth2AuthenticationToken oAuth2AuthenticationToken = (OAuth2AuthenticationToken) authentication;
        if("github".equals(oAuth2AuthenticationToken.getAuthorizedClientRegistrationId())){
            System.out.println("----------------- new users GITHUB");
            DefaultOAuth2User principal = (DefaultOAuth2User) authentication.getPrincipal();
            System.out.println("----------------- new users2");
            Map<String, Object> attributes = principal.getAttributes();
            System.out.println("----------------- new users3" + attributes.toString());
            String email = attributes.getOrDefault("login", "").toString();
            System.out.println("----------------- new users4");
            String name = attributes.getOrDefault("login", "").toString();
        
            userRepo.findByEmail(email).ifPresentOrElse(user -> {
                DefaultOAuth2User newUser = new DefaultOAuth2User(List.of(new SimpleGrantedAuthority(user.getRole().name())), attributes, "id");

                Authentication securityAuth = new OAuth2AuthenticationToken(newUser, List.of(new SimpleGrantedAuthority(user.getRole().name())), oAuth2AuthenticationToken.getAuthorizedClientRegistrationId());

                SecurityContextHolder.getContext().setAuthentication(securityAuth);

            }, () -> {
                UserDTO userToAdd = new UserDTO();
                System.out.println("----------------- new users");
                userToAdd.setRole(UserRole.ROLE_USER);
                userToAdd.setEmail(email);
                userToAdd.setName(name);
                userToAdd.setSource(RegistrationSource.GITHUB);
                userRepo.save(userToAdd);
                
                DefaultOAuth2User newUser = new DefaultOAuth2User(List.of(new SimpleGrantedAuthority(userToAdd.getRole().name())), attributes, "id");

                Authentication securityAuth = new OAuth2AuthenticationToken(newUser, List.of(new SimpleGrantedAuthority(userToAdd.getRole().name())), oAuth2AuthenticationToken.getAuthorizedClientRegistrationId());

                SecurityContextHolder.getContext().setAuthentication(securityAuth);



            });;
        }

        if("google".equals(oAuth2AuthenticationToken.getAuthorizedClientRegistrationId())){
            System.out.println("----------------- new users GOOGLE");
            DefaultOAuth2User principal = (DefaultOAuth2User) authentication.getPrincipal();
            System.out.println("----------------- new users2");
            Map<String, Object> attributes = principal.getAttributes();
            System.out.println("----------------- new users3" + attributes.toString());
            String email = attributes.getOrDefault("email", "").toString();
            System.out.println("----------------- new users4");
            String name = attributes.getOrDefault("name", "").toString();
        
            userRepo.findByEmail(email).ifPresentOrElse(user -> {
                DefaultOAuth2User newUser = new DefaultOAuth2User(List.of(new SimpleGrantedAuthority(user.getRole().name())), attributes, "email");

                Authentication securityAuth = new OAuth2AuthenticationToken(newUser, List.of(new SimpleGrantedAuthority(user.getRole().name())), oAuth2AuthenticationToken.getAuthorizedClientRegistrationId());

                SecurityContextHolder.getContext().setAuthentication(securityAuth);

            }, () -> {
                UserDTO userToAdd = new UserDTO();
                System.out.println("----------------- new users");
                userToAdd.setRole(UserRole.ROLE_USER);
                userToAdd.setEmail(email);
                userToAdd.setName(name);
                userToAdd.setSource(RegistrationSource.GOOGLE);
                userRepo.save(userToAdd);
                
                DefaultOAuth2User newUser = new DefaultOAuth2User(List.of(new SimpleGrantedAuthority(userToAdd.getRole().name())), attributes, "id");

                Authentication securityAuth = new OAuth2AuthenticationToken(newUser, List.of(new SimpleGrantedAuthority(userToAdd.getRole().name())), oAuth2AuthenticationToken.getAuthorizedClientRegistrationId());

                SecurityContextHolder.getContext().setAuthentication(securityAuth);



            });;
        }
        this.setAlwaysUseDefaultTargetUrl(true);
        this.setDefaultTargetUrl("http://localhost:5173/");
        super.onAuthenticationSuccess(request, response, authentication);

    }
}


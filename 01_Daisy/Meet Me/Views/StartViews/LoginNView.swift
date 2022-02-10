//
//  LoginNView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 10.03.21.
//

import SwiftUI
import PromiseKit
import Combine

struct LoginNView: View {
    
    @StateObject var loginVM: LoginViewModel = LoginViewModel()
    @StateObject var developerManager = DeveloperManager()
    
    @Binding var nextPosition: StartPosition
    @Binding var startUpDone: Bool
    @Binding var userToken: String
    
    @State var passwordField1WasEdited: Bool = false
    @State var showPresentTermsAndConditionsSheet: Bool = false
    @State var loginMode: Bool = false
    @State var isChecked: Bool = false
    @State var tryAgain: Bool = false
    
    @Environment(\.colorScheme) var colorScheme
    
    var body: some View {
        
        
        GeometryReader { bounds in
            ZStack {
                
                
                VStack {
                    
                    // MARK: Header
                    VStack(spacing: 0.0) {
                        if !loginMode {
                            Text("Register now for ")
                        }
                        Text("DAISY")
                            .foregroundColor(.accentColor)
                    }
                    .font(.title)
                    
                    // MARK: Terms and Conditions
                    Button(action: {
                        showPresentTermsAndConditionsSheet.toggle()
                    }, label: {
                        HStack {
                            Image(systemName: isChecked ? "checkmark.square" : "square")
                                .onTapGesture {
                                    isChecked.toggle()
                                }
                            Text("By continuing, you agree to our End user licence agreement (EULA) and data policies. Tap here to see them.")
                                .font(.caption)
                                .foregroundColor(.primary)
                        }
                    })
                    .sheet(isPresented: $showPresentTermsAndConditionsSheet) {
                        
                        NavigationView {
                            
                            VStack(alignment: .leading, spacing: 10.0) {
                                Link("Datenschutzerklärung", destination: URL(string: "https://support.berrybblog.com/legal/Datenschutz.pdf")!)
                                Link("Nutzungsbedingungen (EULA)", destination: URL(string: "https://support.berrybblog.com/legal/Nutzungsbedingungen.pdf")!)
                            }
                            .font(.title2)
                            
                            .navigationBarItems(leading: Button("Cancel",
                                                                action: {
                                                                    showPresentTermsAndConditionsSheet = false
                                                                }),
                                                trailing: Button("Done",
                                                                 action: {
                                                                    showPresentTermsAndConditionsSheet = false
                                                                 }))
                        }
                        
                    }
                    
                    // MARK: Error-message can be presented here
                    if loginVM.errorMessage != "" {
                        Text(loginVM.errorMessage)
                            .foregroundColor(.accentColor)
                            .font(.caption)
                            .scaleEffect(tryAgain ? 1.2 : 1)
                            .animation(.spring())
                        
                    }
                    
                    // MARK: Input for Registration
                    TextField("E-Mail", text: $loginVM.email)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .keyboardType(.emailAddress)
                    
                    // password field
                    SecureField("Password", text: $loginVM.password)
                        .onTapGesture {
                            withAnimation(.easeInOut) {
                                passwordField1WasEdited = true
                            }
                        }
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                    
                    // confirm password field
                    if passwordField1WasEdited && !loginMode {
                        SecureField("Repeat password", text: $loginVM.password2)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                    }
                    
                    // token field (only available when login)
                    if !loginMode {
                        TextField("Token", text: $loginVM.loginToken)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .disableAutocorrection(true)
                            .onReceive(Just(loginVM.loginToken)) { _ in limitText(6) }
                        
                    }
                    
                    if !loginMode {
                        Link("No TOKEN? Get infos here", destination: URL(string: "https://support.berrybblog.com")!)
                            .font(.caption)
                            .padding(.bottom, 3)
                    }
                    
                    // MARK: Button to switch between login and register mode
                    Button(action: {
                        withAnimation(.easeInOut) {
                            loginMode.toggle()
                        }
                    }, label: {
                        if !(colorScheme == .dark) {
                            Text(loginMode ? "Don't have an account? Create one" : "I already have an account")
                                .font(.caption)
                                .gradientForeground(gradient: secondaryGradient)
                        } else {
                            Text(loginMode ? "Don't have an account? Create one" : "I already have an account")
                                .font(.caption)
                        }
                        
                        
                    })
                    
                    // MARK: Button to Login/register
                    HStack {
                        Button(action: {
                            
                            if loginVM.errorMessage != "" {
                                tryAgain = true
                                
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                                    tryAgain = false
                                }
                            }
                            
                            // check if the user checked the EULA and data policies
                            if !isChecked {
                                loginVM.errorMessage = "Please accept our EULA and data policies"
                            } else {
                                // if button shows login
                                if loginMode {
                                    print("button tapped")
                                    DispatchQueue.main.async {
                                        firstly{
                                            loginVM.loginAuth()
                                        }.then {
                                            loginVM.checkUserAcc()
                                        }.done { startProcessDone in
                                            if startProcessDone {
                                                self.startUpDone = true
                                            }else {
                                                self.nextPosition = .profileCreation
                                            }
                                        }.catch { error in
                                            loginVM.errorMessage = error.localizedDescription
                                        }
                                    }
                                    
                                }
                                
                                // if button shows register
                                else {
                                    registerUser()
                                }
                            }
                            
                            
                            
                            
                        }, label: {
                            Text(loginMode ? "Login" : "Register")
                                .padding(.horizontal, 50)
                                .padding(.vertical, 8)
                        })
                    }
                }.onAppear {
                    developerManager.getLegalModel().done {
                        print(developerManager.legalModel)
                    }.catch { error in
                        print(error)
                    }
                }
                .frame(maxWidth: .infinity)
                .padding()
                .modifier(offWhiteShadow(cornerRadius: 18))
                .padding(.horizontal, 44)
                
                
                // MARK: - Top-layer
                ZStack {
                    
                    // MARK: Background behind the Symbol
                    BlurView(style: .systemThinMaterial)
                        .frame(width: 71, height: 71)
                        .overlay(
                            Circle()
                                .stroke(
                                    LinearGradient(
                                        gradient: Gradient(stops: [
                                                            .init(color: Color(#colorLiteral(red: 0.7791666388511658, green: 0.7791666388511658, blue: 0.7791666388511658, alpha: 0.949999988079071)), location: 0),
                                                            .init(color: Color(#colorLiteral(red: 0.7250000238418579, green: 0.7250000238418579, blue: 0.7250000238418579, alpha: 0)), location: 1)]),
                                        startPoint: UnitPoint(x: 0.9016393067273221, y: 0.10416647788375455),
                                        endPoint: UnitPoint(x: 0.035519096038869824, y: 0.85416653880629)),
                                    lineWidth: 0.5
                                )
                        )
                        .clipShape(
                            Circle()
                        )
                    
                    // MARK: Symbol which shows the eventStatus
                    Image(systemName: loginMode ? "person.fill" : "person.fill.badge.plus")
                        .font(.system(size: 30))
                        .foregroundColor(.accentColor)
                }
                .offset(x: bounds.size.width / 2 - 44, y: -bounds.size.height / 2 + bounds.size.height * 0.33)
                // when the circle is tapped the login/register mode is toggled
                .onTapGesture {
                    withAnimation(.easeInOut) {
                        loginMode.toggle()
                    }
                }
                
            }
            .padding(.top, bounds.size.height * 0.33)
        }
        
    }
    
    // MARK: Functions
    
    // Function to register the user in the db
    func registerUser() {
        firstly {
            loginVM.checkLoginToken()
        }.done { loginTokenExist in
            
            // pass token to the next screen
            userToken = loginVM.loginToken
            
            if loginTokenExist {
                self.loginVM.register().done {
                    print("viewDone")
                    // switches view to the profile Creation
                    self.nextPosition = .profileCreation
                }.catch { error in
                    loginVM.errorMessage = error.localizedDescription
                }
            } else {
                loginVM.errorMessage = "Token doesn't exist"
            }
        }.catch { error in
            loginVM.errorMessage = error.localizedDescription
            
        }
        
    }
    
    //Function to keep text length in limits
    func limitText(_ upper: Int) {
        if loginVM.loginToken.count > upper {
            loginVM.loginToken = String(loginVM.loginToken.prefix(upper))
        }
    }
}

struct LoginNView_Previews: PreviewProvider {
    static var previews: some View {
        LoginNView(nextPosition: .constant(.registerLogin), startUpDone: .constant(true), userToken: .constant(""))
    }
}

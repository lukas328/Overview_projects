//
//  LoginViewModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 20.01.21.
//

import Foundation
import Firebase
import PromiseKit

class LoginViewModel: ObservableObject {
    
    private var db: Firestore = Firestore.firestore()
    private var firestoreManagerUserTest: FirestoreManagerUserTest = FirestoreManagerUserTest()
    
    @Published var email: String = ""
    @Published var password: String = ""
    @Published var password2: String = ""
    @Published var errorMessage: String = ""
    @Published var loginToken: String = ""

    
    
    
    
    
    
    
    
//    func checkLoginToken() -> Promise<Void>{
//        return Promise { seal in
//
//        firstly {
//            getAllLoginTokens()
//        }.done { tokenCheck in
//            self.checkToken = tokenCheck
//            seal.fulfill(<#T##value: Void##Void#>)
//        }.catch { error in
//            print(error)
//        }
//    }
//    }
    
  
    //login Functions
    func checkLoginToken() ->Promise<Bool> {
        return Promise { seal in
            let _ = db.collection("loginTokens").document("loginTokens").getDocument { snapshot, error in
                if let error = error {
                    seal.reject(error)
                } else {
                    if let snapshot = snapshot {
                        let tokens = try? snapshot.data(as: LoginTokenModel.self)
                        if tokens != nil {
                            DispatchQueue.main.async {
                                if tokens!.tokens.contains(self.loginToken) {
                                    seal.fulfill(true)
                                } else {
                                    seal.fulfill(false)
                                }
                            }
                        }else {
                            seal.reject(Err("no tokens avaliable"))
                        }
                    }
                }
            }
        }
    }
        

    
    func checkUserAcc() -> Promise<Bool> {
        return Promise { seal in
            firstly {
                self.firestoreManagerUserTest.getCurrentUser()
            }.done { userModel in
                seal.fulfill(true)
            }.catch { error in
                seal.fulfill(false)
            }
        }
    }
    
    func loginAuth() ->Promise<Void>{
        return Promise { seal in
            Auth.auth().signIn(withEmail: email, password: password) { (result, error) in
                if let error = error {
                    seal.reject(error)
                } else {
                    seal.fulfill(())
                }
            }
            
        }
        
    }
    
    
    //register Functions 
    
    func register() -> Promise<Void> {
        return Promise{ seal in
            if self.password == self.password2 {
                Auth.auth().createUser(withEmail: email, password: password) { (result, error) in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        seal.fulfill(())
                    }
                }
            }else {
                seal.reject(Err("passwords are not the same"))
            }
        }
    }
}
    




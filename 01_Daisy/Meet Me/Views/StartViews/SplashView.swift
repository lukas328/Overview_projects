//
//  SplashView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 10.03.21.
//

import SwiftUI

struct SplashView: View {
    
    @Binding var currentPosition: StartPosition
    
    @State var showSecondLine: Bool = false
    
    let notchPhone: Bool = UIApplication.shared.windows[0].safeAreaInsets.bottom > 0 ? true : false
    
    var body: some View {
        
        VStack {
            
            Spacer()
            
            VStack {
                HStack(spacing: 0) {
                    Text("Hello ")
                    Text("Nameless")
                        .gradientForeground(gradient: secondaryGradient)
                    Text(",")
                }
                .font(.largeTitle)
                
                
                if showSecondLine {
                    Text("Do you want to meet new people?")
                        .font(.headline)
                }
                
            }
            .padding()
            .modifier(offWhiteShadow(cornerRadius: 12))
            
            Spacer()
            
            Button(action: {
                self.currentPosition = .onboarding
            }, label: {
                button
                    .padding(.bottom, 40)
            })
            .offset(y: notchPhone ? 0 : -55)
        }
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                withAnimation {
                    showSecondLine = true
                }
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            
            self.currentPosition = .onboarding
    }
    }
    
    
    // MARK: -
    var button: some View {
        
        VStack(spacing: 0.0) {
            Text("YES")
                .font(.system(size: 30))
                .foregroundColor(.accentColor)
            Capsule()
                .gradientForeground(gradient: secondaryGradient)
                .frame(width: 58, height: 6)
            
        }
        .frame(maxWidth: .infinity)
        .frame(height: 53)
        .modifier(offWhiteShadow(cornerRadius: 14))
        .padding(.horizontal, 24)
        
        
    }
}

struct SplashView_Previews: PreviewProvider {
    static var previews: some View {
        SplashView(currentPosition: .constant(.splash))
    }
}

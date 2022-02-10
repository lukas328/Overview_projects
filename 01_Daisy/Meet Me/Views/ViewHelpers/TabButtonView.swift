//
//  TabButtonView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 07.03.21.
//

import SwiftUI

struct TabButtonView: View {
    
    @Binding var selectedTab: Tab
    
    var tab: Tab
    var imageName: String
    var animation: Namespace.ID
    
    let gradientMarker = LinearGradient(
        gradient: Gradient(stops: [
                            .init(color: Color(#colorLiteral(red: 0.0313725508749485, green: 0.45490196347236633, blue: 0.5490196347236633, alpha: 1)), location: 0),
                            .init(color: Color(#colorLiteral(red: 0.0313725471496582, green: 0.4549018144607544, blue: 0.5490196347236633, alpha: 0.6299999952316284)), location: 1)]),
        startPoint: UnitPoint(x: 0.9999999999999999, y: 7.105427357601002e-15),
        endPoint: UnitPoint(x: -2.220446049250313e-16, y: -1.7763568394002505e-15)
    )
    
    var body: some View {
        
        // button to witch to the tab
        Button(action: {
//            withAnimation {
                selectedTab = tab
//            }
        }, label: {
            VStack {
                
                // icon of each tab
                Image(systemName: imageName)
                    .font(.system(size: 30))
                    .foregroundColor(selectedTab == tab ? .accentColor : Color(.systemGray3))
                
                // marker which is the current tab
                ZStack {
                    MarkerShape()
                        .fill(Color.clear)
                        .frame(width: 30, height: 6)
                    
                    if selectedTab == tab {
                        MarkerShape()
                            .fill(gradientMarker)
                            .frame(width: 30, height: 6)
                            .matchedGeometryEffect(id: "Tab_Change", in: animation)
                    }
                }
                .animation(.default)
            }
        })
    }
}

struct MarkerShape: Shape {
    
    func path(in rect: CGRect) -> Path {
        
        let path = UIBezierPath(roundedRect: rect, byRoundingCorners: [.bottomLeft, .topRight], cornerRadii: CGSize(width: 10, height: 10))
        
        return Path(path.cgPath)
    }
}
